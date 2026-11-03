"""HTTP API for the web app. Standard library only.

    python3 server.py                      # http://127.0.0.1:8790
    RAG_STORE=memory python3 server.py     # index VAULT_PATH in memory, no Postgres

Endpoints
    POST /ask            {"question": "...", "history": [...]} -> NDJSON stream
    GET  /search?q=...   ranked chunks, no LLM call
    GET  /notes/<path>   a note's text, frontmatter, links and backlinks
    GET  /status         provider, index size, last ingest, backend health
    POST /ingest         re-index the vault in the background

Anything else is served from web/dist when it has been built.
"""

import argparse
import json
import mimetypes
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from rag.config import PROVIDERS, Settings, load_settings
from rag.notes import extract_links, note_name, parse_frontmatter, read_note
from rag.pipeline import Timer, ingest_vault, markdown_files, in_dot_folder, retrieve, stream_answer
from rag.providers import ProviderUnavailable, make_provider
from rag.store import make_store

WEB_DIST = Path(__file__).parent / "web" / "dist"
SNIPPET_CHARS = 280
PING_CACHE_SECONDS = 10
MAX_SEARCH_K = 50
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class NotFound(Exception):
    pass


class App:
    def __init__(self, settings: Settings, store=None, provider=None):
        self.settings = settings
        self.root = settings.vault_root
        self.provider = provider or make_provider(settings)
        self.store = store or make_store(settings)
        self.store.setup()
        self._names: dict[str, str] | None = None
        self._ping = (0.0, None)
        self.indexing = False
        self.index_error = None
        self._ingest_lock = threading.Lock()

    # ---------------------------------------------------------- helpers

    def rel(self, stored_path) -> str | None:
        """Vault-relative path ("Kitchen/Weekend Loaf.md") for a stored path."""
        try:
            return Path(stored_path).relative_to(self.root).as_posix()
        except (ValueError, TypeError):
            return None

    def obsidian_url(self, rel_path) -> str:
        target = rel_path[:-3] if rel_path.lower().endswith(".md") else rel_path
        return f"obsidian://open?vault={quote(self.settings.vault_name, safe='')}&file={quote(target, safe='')}"

    def names(self) -> dict[str, str]:
        """Lower-cased note name -> relative path, for resolving wikilinks."""
        if self._names is None and self.root:
            names = {}
            for f in markdown_files(self.root):
                names.setdefault(note_name(f).lower(), f.relative_to(self.root).as_posix())
            self._names = names
        return self._names or {}

    def resolve_links(self, link_names) -> list[dict]:
        names = self.names()
        return [{"name": n, "path": names.get(n.lower())} for n in link_names]

    def backlinks(self, rel_path) -> list[dict]:
        out = []
        for p in self.store.backlinks_to(note_name(rel_path)):
            rel = self.rel(p)
            if rel and rel != rel_path:
                out.append({"name": note_name(rel), "path": rel})
        return out

    def ping(self) -> dict:
        checked, result = self._ping
        if result is None or time.monotonic() - checked > PING_CACHE_SECONDS:
            result = self.provider.ping()
            self._ping = (time.monotonic(), result)
        return result

    def require_backend(self):
        """Fail fast when pc is down. Without this, an unreachable pc makes the
        embed call wait out the full read timeout (twice) before erroring."""
        if self.settings.provider != "local":
            return {}
        health = self.ping()
        if not health.get("reachable"):
            raise ProviderUnavailable(health.get("error") or "Can't reach Ollama.")
        return health

    # ---------------------------------------------------------- endpoints

    def status(self) -> dict:
        spec = self.settings.spec()
        return {
            "provider": spec.name,
            "providers": list(PROVIDERS),
            "embed_model": spec.embed_model,
            "chat_model": spec.chat_model,
            "store": self.settings.store,
            "table": self.store.table,
            "vault_name": self.settings.vault_name,
            "index": self.store.stats(),
            "last_ingest": self.store.last_ingest(),
            "indexing": self.indexing,
            "index_error": self.index_error,
            "backend": self.ping(),
        }

    def source(self, chunk, n) -> dict:
        rel = self.rel(chunk["file"]) or Path(chunk["file"]).name
        text = snippet_text(chunk["content"], note_name(rel))
        snippet = text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
        return {
            "n": n,
            "path": rel,
            "title": note_name(rel),
            "heading": chunk.get("heading"),
            "score": None if chunk.get("via_link") else round(chunk["similarity"], 4),
            "via_link": bool(chunk.get("via_link")),
            "via": chunk.get("via"),
            "snippet": snippet,
            "obsidian_url": self.obsidian_url(rel),
            "links": self.resolve_links(chunk.get("links", [])),
            "backlinks": self.backlinks(rel),
        }

    def search(self, query, top_k=None) -> dict:
        timer = Timer()
        self.require_backend()
        chunks = retrieve(self.store, self.provider, query, top_k or self.settings.top_k,
                          self.settings.threshold, **self.settings.retrieval_options())
        return {"query": query, "results": [self.source(c, i + 1) for i, c in enumerate(chunks)],
                "ms": timer.ms()}

    def note(self, rel_path) -> dict:
        if not self.root:
            raise NotFound("No vault configured")
        root = self.root.resolve()
        path = (root / rel_path).resolve()
        if (not path.is_relative_to(root) or path.suffix.lower() != ".md"
                or in_dot_folder(path, root) or not path.is_file()):
            raise NotFound(rel_path)
        rel = path.relative_to(root).as_posix()
        meta, body = parse_frontmatter(read_note(path))
        stored = str(self.root / rel)
        return {
            "path": rel,
            "title": note_name(rel),
            "frontmatter": meta,
            "body": body,
            "obsidian_url": self.obsidian_url(rel),
            "links": self.resolve_links(extract_links(body)),
            "backlinks": self.backlinks(rel),
            "indexed": self.store.file_hash(stored) is not None,
        }

    def ask(self, question, history=None):
        """Yield NDJSON-ready events: status?, sources, token*, done | error."""
        timer = Timer()
        try:
            if self.settings.provider == "local":
                health = self.require_backend()
                if not health.get("chat_loaded"):
                    yield {"type": "status", "message":
                           f"Loading {self.settings.ollama_chat_model} on pc. "
                           "The first answer after a pause takes a few extra seconds."}
            chunks = retrieve(self.store, self.provider, question, self.settings.top_k,
                              self.settings.threshold, **self.settings.retrieval_options())
            yield {"type": "sources", "sources": [self.source(c, i + 1) for i, c in enumerate(chunks)],
                   "retrieval_ms": timer.ms()}
            first = None
            for token in stream_answer(self.provider, question, chunks, history):
                if first is None:
                    first = timer.ms()
                yield {"type": "token", "text": token}
            self._ping = (0.0, None)  # models are warm now; refresh status next time
            yield {"type": "done", "first_token_ms": first, "total_ms": timer.ms()}
        except ProviderUnavailable as e:
            self._ping = (0.0, None)
            yield {"type": "error", "code": "backend_unavailable", "message": str(e)}
        except Exception as e:  # headers are sent already, so report it in the stream
            yield {"type": "error", "code": "internal", "message": f"Something broke on the server: {e}"}

    def ingest(self):
        if not self.root:
            self.index_error = "Set VAULT_PATH to index a vault."
            return
        if not self._ingest_lock.acquire(blocking=False):
            return  # already running
        self.indexing = True
        self.index_error = None
        try:
            ingest_vault(self.store, self.provider, self.root, log=lambda *_: None,
                         chunking=self.settings.chunking)
            self._names = None
        except Exception as e:
            self.index_error = str(e)
        finally:
            self.indexing = False
            self._ingest_lock.release()

    def ingest_in_background(self):
        threading.Thread(target=self.ingest, daemon=True).start()


def snippet_text(content: str, title: str) -> str:
    """Drop the "Tags: ..." line and the repeated note title from a chunk's start."""
    if content.startswith("Tags: ") and f" {title}" in content:
        content = content.split(f" {title}", 1)[1].lstrip()
    elif content.startswith(title + " "):
        content = content[len(title):].lstrip()
    return content


class Handler(BaseHTTPRequestHandler):
    app: App = None  # set by make_server
    allowed_hosts = LOOPBACK_HOSTS
    server_version = "obsidian-rag"

    def log_message(self, fmt, *args):  # quieter than the default
        pass

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._json({"error": message}, status)

    def _host_allowed(self) -> bool:
        """Reject requests whose Host isn't this server, so a web page that
        rebinds its domain to 127.0.0.1 (DNS rebinding) can't read notes."""
        host = self.headers.get("Host")
        if host is None:
            return True  # HTTP/1.0 clients; browsers always send Host
        name = urlparse("//" + host).hostname or ""
        return name in self.allowed_hosts

    def _refuse(self):
        self._error(HTTPStatus.FORBIDDEN,
                    "Host not allowed. Add it to RAG_ALLOWED_HOSTS to reach the server by that name.")

    def do_GET(self):
        if not self._host_allowed():
            return self._refuse()
        url = urlparse(self.path)
        try:
            if url.path == "/status":
                return self._json(self.app.status())
            if url.path == "/search":
                params = parse_qs(url.query)
                query = (params.get("q") or [""])[0].strip()
                if not query:
                    return self._error(HTTPStatus.BAD_REQUEST, "Add a query: /search?q=...")
                k = int((params.get("k") or [0])[0]) or None
                if k is not None and not 1 <= k <= MAX_SEARCH_K:
                    return self._error(HTTPStatus.BAD_REQUEST, f"k must be between 1 and {MAX_SEARCH_K}.")
                return self._json(self.app.search(query, k))
            if url.path.startswith("/notes/"):
                return self._json(self.app.note(unquote(url.path[len("/notes/"):])))
            return self._static(url.path)
        except NotFound:
            return self._error(HTTPStatus.NOT_FOUND, "Note not found")
        except ProviderUnavailable as e:
            return self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(e))
        except ValueError as e:
            return self._error(HTTPStatus.BAD_REQUEST, str(e))

    def do_POST(self):
        if not self._host_allowed():
            return self._refuse()
        # Other sites can send text/plain or form POSTs here without a CORS
        # preflight; requiring JSON means a browser only sends same-origin ones.
        content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if content_type != "application/json":
            return self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Send Content-Type: application/json.")
        url = urlparse(self.path)
        if url.path == "/ingest":
            self.app.ingest_in_background()
            return self._json({"started": True}, HTTPStatus.ACCEPTED)
        if url.path != "/ask":
            return self._error(HTTPStatus.NOT_FOUND, "Not found")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            body = None
        if not isinstance(body, dict):
            return self._error(HTTPStatus.BAD_REQUEST, "Send JSON: {\"question\": \"...\"}")
        question = str(body.get("question") or "").strip()
        if not question:
            return self._error(HTTPStatus.BAD_REQUEST, "Ask a question first.")
        history = body.get("history")
        history = [m for m in (history if isinstance(history, list) else [])
                   if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                   and isinstance(m.get("content"), str) and m["content"].strip()]

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        events = self.app.ask(question, history)
        try:
            for event in events:
                self.wfile.write((json.dumps(event) + "\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser went away; stop generating
        finally:
            events.close()

    def _static(self, path):
        if not WEB_DIST.is_dir():
            return self._error(HTTPStatus.NOT_FOUND, "Not found (build the web app with `npm run build` in web/)")
        target = (WEB_DIST / path.lstrip("/")).resolve()
        if not target.is_relative_to(WEB_DIST.resolve()) or not target.is_file():
            target = WEB_DIST / "index.html"  # single-page app fallback
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(app: App, host="127.0.0.1", port=8790) -> ThreadingHTTPServer:
    extra = {h.strip().lower() for h in os.getenv("RAG_ALLOWED_HOSTS", "").split(",") if h.strip()}
    bind = host.strip("[]").lower()
    allowed = LOOPBACK_HOSTS | extra | ({bind} if bind not in ("", "0.0.0.0", "::") else set())
    handler = type("BoundHandler", (Handler,), {"app": app, "allowed_hosts": allowed})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description="Serve the Obsidian RAG API and web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--provider", choices=PROVIDERS)
    parser.add_argument("--vault")
    parser.add_argument("--memory", action="store_true", help="Same as RAG_STORE=memory")
    args = parser.parse_args()

    overrides = {k: v for k, v in (("provider", args.provider), ("vault_path", args.vault)) if v}
    if args.memory:
        overrides["store"] = "memory"
    app = App(load_settings(**overrides))
    if app.settings.store == "memory":
        app.ingest_in_background()
    server = make_server(app, args.host, args.port)
    print(f"Serving on http://{args.host}:{args.port} ({app.settings.provider} provider, "
          f"{app.settings.store} store)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.store.close()


if __name__ == "__main__":
    main()
