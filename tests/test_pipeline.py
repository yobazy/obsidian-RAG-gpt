import shutil
import tempfile
import unittest
from pathlib import Path

from rag.config import Settings, load_settings
from rag.pipeline import answer, ingest_vault, retrieve
from rag.providers import strip_think
from rag.store import MemoryStore
from tests.fakes import FIXTURE_VAULT, FakeProvider


def quiet(*_):
    pass


class IngestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.vault = self.tmp / "vault"
        shutil.copytree(FIXTURE_VAULT, self.vault)
        self.store = MemoryStore()
        self.provider = FakeProvider()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_ingest_skips_dot_folders_and_tiny_notes(self):
        summary = ingest_vault(self.store, self.provider, self.vault, log=quiet)
        names = {Path(p).name for p in self.store.file_paths()}
        self.assertNotIn("Ignored.md", names)
        self.assertNotIn("Inbox.md", names)
        self.assertIn("Weekend Loaf.md", names)
        self.assertEqual(summary["files"], 27)
        self.assertEqual(summary["changed"], 26)

    def test_reingest_is_hash_based(self):
        ingest_vault(self.store, self.provider, self.vault, log=quiet)
        calls = self.provider.embed_calls
        summary = ingest_vault(self.store, self.provider, self.vault, log=quiet)
        self.assertEqual(summary["chunks"], 0)
        self.assertEqual(self.provider.embed_calls, calls)

        note = self.vault / "Garden Log.md"
        note.write_text(note.read_text() + "\n## Beans\nRunner beans went in on 1 June.\n")
        summary = ingest_vault(self.store, self.provider, self.vault, log=quiet)
        self.assertEqual(summary["changed"], 1)

    def test_deleted_notes_leave_the_index(self):
        ingest_vault(self.store, self.provider, self.vault, log=quiet)
        (self.vault / "Garden Log.md").unlink()
        summary = ingest_vault(self.store, self.provider, self.vault, log=quiet)
        self.assertEqual(summary["removed"], 1)
        self.assertFalse(any(p.endswith("Garden Log.md") for p in self.store.file_paths()))

    def test_other_vaults_rows_are_kept(self):
        self.store.replace_file("/elsewhere/Other.md", "h", [], [], [])
        ingest_vault(self.store, self.provider, self.vault, log=quiet)
        self.assertIn("/elsewhere/Other.md", self.store.file_paths())

    def test_one_bad_note_does_not_stop_the_ingest(self):
        class Picky(FakeProvider):
            def embed(self, texts, kind):
                if any("Bramble" in t for t in texts):
                    raise RuntimeError("bad note")
                return super().embed(texts, kind)

        summary = ingest_vault(self.store, Picky(), self.vault, log=quiet)
        self.assertEqual(summary["failed"], ["Sourdough Starter.md"])
        self.assertEqual(summary["changed"], 25)
        self.assertIsNotNone(self.store.last_ingest())

    def test_last_ingest_is_logged(self):
        ingest_vault(self.store, self.provider, self.vault, log=quiet)
        self.assertEqual(self.store.last_ingest()["files_seen"], 27)


class RetrieveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = MemoryStore()
        cls.provider = FakeProvider()
        ingest_vault(cls.store, cls.provider, FIXTURE_VAULT, log=quiet)

    def test_top_hit_and_heading(self):
        chunks = retrieve(self.store, self.provider, "weekend loaf hydration percent", min_similarity=0.1)
        self.assertTrue(chunks[0]["file"].endswith("Weekend Loaf.md"))
        self.assertFalse(chunks[0]["via_link"])

    def test_links_expand_context(self):
        chunks = retrieve(self.store, self.provider, "station code NB04 harbour CSV timestamps", top_k=1, min_similarity=0.1)
        files = [Path(c["file"]).name for c in chunks]
        self.assertEqual(files[0], "Tide Data Source.md")
        self.assertIn("Lighthouse App.md", files)
        linked = [c for c in chunks if c["via_link"]]
        self.assertTrue(all(c["similarity"] == 0.0 for c in linked))

    def test_threshold_filters_everything(self):
        self.assertEqual(retrieve(self.store, self.provider, "quantum chromodynamics", min_similarity=0.9), [])

    def test_backlinks(self):
        sources = {Path(p).name for p in self.store.backlinks_to("sourdough starter")}
        self.assertEqual(sources, {"Weekend Loaf.md", "Pantry Staples.md", "Focaccia.md"})

    def test_answer_keeps_history(self):
        history = []
        chunks = retrieve(self.store, self.provider, "starter feeding", min_similarity=0.1)
        text = answer(self.provider, "starter feeding", chunks, history)
        self.assertIn("Bramble", text)
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        self.assertIn("Sourdough Starter.md", self.provider.chat_messages[-1]["content"])


class ThinkTest(unittest.TestCase):
    def test_think_block_is_removed_across_tokens(self):
        tokens = ["<thi", "nk>\nhmm", " let me see</th", "ink>\n\nThe", " answer"]
        self.assertEqual("".join(strip_think(iter(tokens))), "The answer")

    def test_plain_text_passes_through(self):
        self.assertEqual("".join(strip_think(iter(["<", "b>bold", " text"]))), "<b>bold text")
        self.assertEqual("".join(strip_think(iter(["Hi", " there"]))), "Hi there")


class SettingsTest(unittest.TestCase):
    def test_providers_use_separate_tables(self):
        cloud = Settings(provider="cloud").spec()
        local = Settings(provider="local").spec()
        self.assertEqual(cloud.table, "chunks")
        self.assertEqual((cloud.dim, local.dim), (1024, 768))
        self.assertNotEqual(cloud.table, local.table)

    def test_unknown_provider_is_rejected(self):
        import os

        old = os.environ.get("RAG_PROVIDER")
        os.environ["RAG_PROVIDER"] = "gpt"
        try:
            with self.assertRaises(ValueError):
                load_settings()
        finally:
            if old is None:
                del os.environ["RAG_PROVIDER"]
            else:
                os.environ["RAG_PROVIDER"] = old


if __name__ == "__main__":
    unittest.main()
