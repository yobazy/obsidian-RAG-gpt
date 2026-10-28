import os
import time
import unittest

from rag.config import Settings
from rag.providers import Ollama, ProviderUnavailable

LIVE = os.getenv("RAG_LIVE_OLLAMA") == "1"


class UnreachableOllamaTest(unittest.TestCase):
    def setUp(self):
        # Port 9 (discard) on localhost refuses connections immediately.
        self.ollama = Ollama(Settings(provider="local", ollama_url="http://127.0.0.1:9", ollama_timeout=3))

    def test_embed_raises_a_friendly_error(self):
        started = time.perf_counter()
        with self.assertRaises(ProviderUnavailable) as ctx:
            self.ollama.embed(["hello"], kind="query")
        self.assertIn("Can't reach Ollama", str(ctx.exception))
        self.assertLess(time.perf_counter() - started, 10)

    def test_ping_reports_down_without_raising(self):
        self.assertFalse(self.ollama.ping(timeout=1)["reachable"])


@unittest.skipUnless(LIVE, "set RAG_LIVE_OLLAMA=1 to call the real Ollama on pc")
class LiveOllamaTest(unittest.TestCase):
    def setUp(self):
        self.ollama = Ollama(Settings(provider="local", ollama_url=os.getenv("OLLAMA_URL", Settings.ollama_url)))

    def test_embedding_is_768_dims(self):
        vectors = self.ollama.embed(["bread flour", "tide tables"], kind="document")
        self.assertEqual([len(v) for v in vectors], [768, 768])

    def test_chat_streams_without_think_tags(self):
        text = "".join(self.ollama.stream_chat("Reply in one short sentence.", [{"role": "user", "content": "Say hello."}]))
        self.assertTrue(text.strip())
        self.assertNotIn("<think>", text)


if __name__ == "__main__":
    unittest.main()
