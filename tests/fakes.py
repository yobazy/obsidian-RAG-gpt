"""Deterministic stand-ins for the embedding and answer providers."""

import hashlib
import re
from pathlib import Path

FIXTURE_VAULT = Path(__file__).parent / "fixtures" / "vault"
DIM = 256


class FakeProvider:
    """Bag-of-words embeddings: texts sharing words get similar vectors."""

    def __init__(self, reply="Bramble is fed weekly [1]."):
        self.reply = reply
        self.embed_calls = 0
        self.chat_messages = None

    def embed(self, texts, kind):
        self.embed_calls += 1
        return [self._vector(t) for t in texts]

    @staticmethod
    def _vector(text):
        vec = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if len(word) < 3:
                continue
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        return vec

    def stream_chat(self, system, messages):
        self.chat_messages = messages
        for word in self.reply.split(" "):
            yield word + " "

    def ping(self, timeout=4.0):
        return {"reachable": True, "embed_loaded": True, "chat_loaded": True}


class DownProvider(FakeProvider):
    def embed(self, texts, kind):
        from rag.providers import ProviderUnavailable

        raise ProviderUnavailable("Can't reach Ollama at https://example.invalid (test).")

    def ping(self, timeout=4.0):
        return {"reachable": False, "error": "Can't reach Ollama (test)."}
