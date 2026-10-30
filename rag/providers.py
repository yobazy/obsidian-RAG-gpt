"""Embedding and answer providers.

local: Ollama's native API (/api/embed, /api/chat), stdlib only.
cloud: Voyage AI + Claude, imported lazily so local mode needs neither SDK.
"""

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Iterator

from .config import Settings


class ProviderUnavailable(RuntimeError):
    """The backend could not be reached or refused the request."""


# ---------------------------------------------------------------- Ollama


class Ollama:
    def __init__(self, settings: Settings):
        self.url = settings.ollama_url
        self.embed_model = settings.ollama_embed_model
        self.chat_model = settings.ollama_chat_model
        self.num_ctx = settings.ollama_num_ctx
        self.timeout = settings.ollama_timeout
        self.connect_timeout = settings.ollama_connect_timeout

    def _open(self, path, body=None, timeout=None, retries=1):
        """POST JSON (or GET when body is None). Retries once on connection
        errors, since a pc that just woke up can drop the first request."""
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        last_error = None
        for attempt in range(retries + 1):
            try:
                return urllib.request.urlopen(request, timeout=timeout or self.timeout)
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="ignore")[:300]
                last_error = ProviderUnavailable(f"Ollama returned {e.code}: {detail}")
                # 5xx is often transient on a shared GPU (model loading, out of memory).
                if e.code >= 500 and attempt < retries:
                    time.sleep(5)
                    continue
                raise last_error from e
            except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as e:
                last_error = e
                if attempt < retries:
                    time.sleep(2)
        raise ProviderUnavailable(
            f"Can't reach Ollama at {self.url} ({_reason(last_error)}). "
            "Check that pc is on and Tailscale is connected."
        ) from last_error

    def ping(self, timeout=4.0) -> dict:
        """Reachability and which models are loaded right now."""
        try:
            with self._open("/api/ps", timeout=timeout, retries=0) as r:
                loaded = [m.get("name", "") for m in json.load(r).get("models", [])]
        except ProviderUnavailable as e:
            return {"reachable": False, "error": str(e)}

        def is_loaded(model):
            full = model if ":" in model else model + ":latest"
            return full in loaded

        return {
            "reachable": True,
            "embed_loaded": is_loaded(self.embed_model),
            "chat_loaded": is_loaded(self.chat_model),
        }

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        # nomic-embed-text is trained with task prefixes; without them
        # query/document similarity drops noticeably.
        prefix = "search_query: " if kind == "query" else "search_document: "
        if "nomic" not in self.embed_model:
            prefix = ""
        vectors = []
        for start in range(0, len(texts), 32):
            batch = [prefix + t for t in texts[start:start + 32]]
            with self._open(
                "/api/embed",
                {"model": self.embed_model, "input": batch, "truncate": True},
            ) as r:
                vectors.extend(json.load(r)["embeddings"])
        return vectors

    def stream_chat(self, system: str, messages: list[dict]) -> Iterator[str]:
        body = {
            "model": self.chat_model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": True,
            "think": False,  # qwen3 would otherwise spend seconds "thinking"
            "options": {"num_ctx": self.num_ctx, "temperature": 0.3},
        }
        with self._open("/api/chat", body) as r:
            yield from strip_think(_ollama_tokens(r))

    def chat_json(self, system: str, prompt: str, schema: dict, timeout=None) -> dict:
        """One non-streamed reply constrained to a JSON schema (used by rerank)."""
        body = {
            "model": self.chat_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"num_ctx": self.num_ctx, "temperature": 0},
        }
        with self._open("/api/chat", body, timeout=timeout, retries=0) as r:
            reply = json.load(r)
        if reply.get("error"):
            raise ProviderUnavailable(f"Ollama: {reply['error']}")
        text = "".join(strip_think(iter([reply.get("message", {}).get("content", "")])))
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ProviderUnavailable("Ollama sent a reply this app can't read.") from e


def _lines(response):
    try:
        yield from response
    except (OSError, socket.timeout) as e:  # dropped connection or read timeout mid-answer
        raise ProviderUnavailable(f"Ollama stopped mid-answer ({_reason(e)}).") from e


def _ollama_tokens(response) -> Iterator[str]:
    for line in _lines(response):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as e:
            raise ProviderUnavailable("Ollama sent a reply this app can't read.") from e
        if event.get("error"):
            raise ProviderUnavailable(f"Ollama: {event['error']}")
        text = event.get("message", {}).get("content", "")
        if text:
            yield text
        if event.get("done"):
            return


def strip_think(tokens: Iterator[str]) -> Iterator[str]:
    """Drop a leading <think>...</think> block if the model emits one anyway."""
    buffer = ""
    in_think = None  # None until we know whether the reply opens with <think>
    for token in tokens:
        if in_think is False:
            yield token
            continue
        buffer += token
        if in_think is None:
            stripped = buffer.lstrip()
            if len(stripped) < len("<think>") and "<think>".startswith(stripped):
                continue  # not enough text yet to decide
            in_think = stripped.startswith("<think>")
            if not in_think:
                yield buffer
                buffer = ""
                continue
        end = buffer.find("</think>")
        if end != -1:
            rest = buffer[end + len("</think>"):].lstrip()
            in_think = False
            buffer = ""
            if rest:
                yield rest
    if in_think is None and buffer:
        yield buffer


def _reason(error) -> str:
    if error is None:
        return "unknown error"
    reason = getattr(error, "reason", error)
    return str(reason) or error.__class__.__name__


# ---------------------------------------------------------------- Cloud


class Cloud:
    def __init__(self, settings: Settings):
        self.embed_model = settings.voyage_model
        self.chat_model = settings.claude_model
        self._voyage = None
        self._claude = None

    @property
    def voyage(self):
        if self._voyage is None:
            import os

            import voyageai

            self._voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        return self._voyage

    @property
    def claude(self):
        if self._claude is None:
            import os

            import anthropic

            self._claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return self._claude

    def ping(self, timeout=4.0) -> dict:
        import os

        missing = [k for k in ("VOYAGE_API_KEY", "ANTHROPIC_API_KEY") if not os.getenv(k)]
        if missing:
            return {"reachable": False, "error": f"Missing {', '.join(missing)}"}
        return {"reachable": True}

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        try:
            result = self.voyage.embed(texts, model=self.embed_model, input_type=kind)
        except Exception as e:  # SDK raises its own error types
            raise ProviderUnavailable(f"Voyage AI: {e}") from e
        return result.embeddings

    def stream_chat(self, system: str, messages: list[dict]) -> Iterator[str]:
        try:
            with self.claude.messages.stream(
                model=self.chat_model,
                max_tokens=1024,
                system=system,
                messages=messages,
            ) as stream:
                yield from stream.text_stream
        except Exception as e:
            raise ProviderUnavailable(f"Claude: {e}") from e


def make_provider(settings: Settings):
    return Ollama(settings) if settings.provider == "local" else Cloud(settings)

