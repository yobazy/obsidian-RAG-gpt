import unittest
from pathlib import Path

from rag.pipeline import ingest_vault, retrieve
from rag.rerank import RERANK_CANDIDATES, rerank_rows
from rag.store import MemoryStore
from tests.fakes import FIXTURE_VAULT, FakeProvider


def rows(n):
    return [{"file": f"/v/N{i}.md", "content": f"text {i}", "heading": None, "chunk_index": 0}
            for i in range(n)]


class Ranker(FakeProvider):
    def __init__(self, reply=None, error=None):
        super().__init__()
        self.reply, self.error, self.prompts = reply, error, []

    def chat_json(self, system, prompt, schema, timeout=None):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.reply


def files(result):
    return [Path(r["file"]).stem for r in result]


class RerankTest(unittest.TestCase):
    def test_reorders_and_keeps_the_rest(self):
        out = rerank_rows(Ranker({"ranking": [3, 1]}), "q", rows(4))
        self.assertEqual(files(out), ["N2", "N0", "N1", "N3"])
        self.assertTrue(out[0]["reranked"])
        self.assertNotIn("reranked", out[2])

    def test_bad_numbers_are_ignored(self):
        out = rerank_rows(Ranker({"ranking": [9, 0, 2, 2, True, "1"]}), "q", rows(3))
        self.assertEqual(files(out), ["N1", "N0", "N2"])

    def test_failures_keep_the_order(self):
        for provider in (Ranker(error=TimeoutError("slow")), Ranker({"nope": 1}), Ranker("junk"),
                         FakeProvider()):
            if isinstance(provider, Ranker):
                with self.assertLogs("rag.rerank", "WARNING"):
                    out = rerank_rows(provider, "q", rows(3))
            else:
                out = rerank_rows(provider, "q", rows(3))
            self.assertEqual(files(out), ["N0", "N1", "N2"])

    def test_only_the_head_is_sent(self):
        ranker = Ranker({"ranking": [2]})
        out = rerank_rows(ranker, "q", rows(RERANK_CANDIDATES + 3))
        self.assertIn(f"[{RERANK_CANDIDATES}]", ranker.prompts[0])
        self.assertNotIn(f"[{RERANK_CANDIDATES + 1}]", ranker.prompts[0])
        self.assertEqual(len(out), RERANK_CANDIDATES + 3)
        self.assertEqual(files(out)[0], "N1")

    def test_long_passages_are_cut(self):
        long = [{"file": "/v/A.md", "content": "word " * 1000, "heading": "Part", "chunk_index": 0},
                {"file": "/v/B.md", "content": "short", "heading": None, "chunk_index": 0}]
        ranker = Ranker({"ranking": [2, 1]})
        rerank_rows(ranker, "q", long)
        self.assertLess(len(ranker.prompts[0]), 1200)
        self.assertIn("[1] A > Part", ranker.prompts[0])

    def test_retrieve_uses_it(self):
        store = MemoryStore()
        ranker = Ranker()
        ingest_vault(store, ranker, FIXTURE_VAULT, log=lambda *_: None)
        plain = retrieve(store, ranker, "starter flour rye", top_k=3, min_similarity=0.0, mode="hybrid")
        primary = [r for r in plain if not r["via_link"]]
        ranker.reply = {"ranking": [3, 2, 1]}
        reranked = retrieve(store, ranker, "starter flour rye", top_k=3, min_similarity=0.0,
                            mode="hybrid", rerank=True)
        self.assertEqual(files([r for r in reranked if not r["via_link"]]),
                         files(primary)[::-1])


if __name__ == "__main__":
    unittest.main()
