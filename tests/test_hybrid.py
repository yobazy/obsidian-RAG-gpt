import shutil
import tempfile
import unittest
from pathlib import Path

from rag.notes import CHUNK_SIZE, chunk_note, matches_filters, note_tags, parse_filters
from rag.pipeline import fuse, index_hash, ingest_vault, merge_filters, retrieve
from rag.store import MemoryStore, PostgresStore, _tsquery_words, keyword_terms
from tests.fakes import FIXTURE_VAULT, FakeProvider


def quiet(*_):
    pass


def names(chunks):
    return [Path(c["file"]).stem for c in chunks]


class ChunkingTest(unittest.TestCase):
    def test_short_note_stays_one_chunk(self):
        text = (FIXTURE_VAULT / "Kitchen" / "Weekend Loaf.md").read_text()
        self.assertEqual(len(chunk_note(text, mode="heading", title="Weekend Loaf")), 1)
        self.assertEqual(len(chunk_note(text, mode="window")), 1)

    def test_long_note_splits_at_headings_with_title_context(self):
        text = (FIXTURE_VAULT / "Travel" / "Highlands Road Trip.md").read_text()
        chunks = chunk_note(text, mode="heading", title="Highlands Road Trip")
        headings = [c.heading for c in chunks]
        self.assertIn("Day five: Tongue to Aviemore", headings)
        day5 = chunks[headings.index("Day five: Tongue to Aviemore")]
        self.assertIn("Rothiemurchus", day5.content)
        self.assertNotIn("migraine", day5.content)  # that's day six
        self.assertTrue(day5.index_text.startswith("Highlands Road Trip > Day five"))
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))

    def test_small_sections_merge_into_the_next(self):
        body = "# Long\n\n" + "\n\n".join(
            f"## Part {i}\n\n" + ("word " * (5 if i == 1 else 100)) for i in range(1, 6))
        chunks = chunk_note(body, mode="heading", title="Long")
        self.assertGreater(sum(len(c.content.split()) for c in chunks), CHUNK_SIZE)
        self.assertNotIn("Part 1", [c.heading for c in chunks[1:]])
        self.assertTrue(all(len(c.content.split()) >= 25 for c in chunks))

    def test_long_section_still_windowed(self):
        body = "# Big\n\n## Only\n\n" + " ".join(f"w{i}" for i in range(1000))
        chunks = chunk_note(body, mode="heading", title="Big")
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(c.content.split()) <= CHUNK_SIZE for c in chunks))

    def test_short_intro_does_not_relabel_the_next_section(self):
        body = "# Loaf\n\nA short intro.\n\n## Lessons\n\n" + "word " * 450
        chunks = chunk_note(body, mode="heading", title="Loaf")
        self.assertEqual(chunks[0].heading, "Lessons")
        self.assertTrue(chunks[0].index_text.startswith("Loaf > Lessons"))

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            chunk_note("# x\n" + "word " * 500, mode="sentences")


class FilterParsingTest(unittest.TestCase):
    def test_inline_filters(self):
        q, f = parse_filters('#book which one tag:Fiction in:"Side Projects" about C# issue #12')
        self.assertEqual(q, "which one about C# issue #12")
        self.assertEqual(f, {"tags": ["book", "fiction"], "folder": "Side Projects"})

    def test_trailing_punctuation_is_not_part_of_the_filter(self):
        self.assertEqual(parse_filters("What recipes are in:Kitchen?"),
                         ("What recipes are?", {"folder": "Kitchen"}))
        self.assertEqual(parse_filters("tag:book, any?")[1], {"tags": ["book"]})
        self.assertEqual(parse_filters("what is #kimchi?")[0], "what is?")

    def test_folder_is_matched_from_the_vault_root(self):
        path = "/home/u/Documents/vault/Kitchen/Kimchi.md"
        self.assertTrue(matches_filters(path, [], {"folder": "Documents"}))  # no root: segment match
        root = "/home/u/Documents/vault"
        self.assertFalse(matches_filters(path, [], {"folder": "Documents", "root": root}))
        self.assertTrue(matches_filters(path, [], {"folder": "kitchen", "root": root}))
        nested = "/home/u/Documents/vault/Projects/Kitchen/Plan.md"
        self.assertFalse(matches_filters(nested, [], {"folder": "Kitchen", "root": root}))

    def test_inline_body_tags(self):
        body = "Notes #recipe and #Bread/sourdough.\n# Heading\n`#notatag`\nissue #12 C#"
        self.assertEqual(note_tags({"tags": ["recipe"]}, body), ["recipe", "bread/sourdough"])

    def test_no_filters_leaves_query_alone(self):
        self.assertEqual(parse_filters("what is 2#3?"), ("what is 2#3?", {}))

    def test_only_filters_keeps_the_query(self):
        q, f = parse_filters("#book")
        self.assertEqual((q, f), ("#book", {"tags": ["book"]}))

    def test_matching(self):
        self.assertTrue(matches_filters("/v/Reading/Piranesi.md", ["book/fiction"], {"tags": ["book"]}))
        self.assertFalse(matches_filters("/v/Reading/Piranesi.md", ["bookish"], {"tags": ["book"]}))
        self.assertTrue(matches_filters("/v/Reading/Piranesi.md", [], {"folder": "reading"}))
        self.assertFalse(matches_filters("/v/Reading/Piranesi.md", [], {"folder": "Read"}))
        self.assertTrue(matches_filters("/v/a.md", [], {}))

    def test_tags_from_frontmatter(self):
        self.assertEqual(note_tags({"tags": ["#Baking", "recipe"]}), ["baking", "recipe"])
        self.assertEqual(note_tags({"tags": "a, b"}), ["a", "b"])
        self.assertEqual(note_tags({}), [])

    def test_merge(self):
        self.assertEqual(merge_filters({"tags": ["a"]}, None, {"tags": ["b", "a"], "folder": "X"}),
                         {"tags": ["a", "b"], "folder": "X"})


class FuseTest(unittest.TestCase):
    def test_rrf_rewards_agreement(self):
        a = [{"file": "x", "chunk_index": 0}, {"file": "y", "chunk_index": 0}]
        b = [{"file": "y", "chunk_index": 0}, {"file": "z", "chunk_index": 0}]
        fused = fuse([a, b])
        self.assertEqual([r["file"] for r in fused], ["y", "x", "z"])
        self.assertAlmostEqual(fused[0]["score"], 1 / 62 + 1 / 61)


class KeywordSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = MemoryStore()
        cls.provider = FakeProvider()
        ingest_vault(cls.store, cls.provider, FIXTURE_VAULT, log=quiet)

    def test_rare_word_ranks_first(self):
        rows = self.store.keyword_search("where do I get gochugaru", 5)
        self.assertEqual(names(rows)[0], "Kimchi")

    def test_stopwords_only(self):
        self.assertEqual(self.store.keyword_search("what is the", 5), [])

    def test_filters_apply(self):
        rows = self.store.keyword_search("tide", 10, {"tags": ["book"]})
        self.assertEqual(set(names(rows)), {"Piranesi"})
        rows = self.store.keyword_search("tide", 10, {"folder": "Projects"})
        self.assertTrue(rows and all("/Projects/" in r["file"] for r in rows))

    def test_index_refreshes_after_changes(self):
        store = MemoryStore()
        ingest_vault(store, self.provider, FIXTURE_VAULT, log=quiet)
        self.assertTrue(store.keyword_search("gochugaru", 5))
        kimchi = next(p for p in store.file_paths() if p.endswith("Kimchi.md"))
        store.delete_file(kimchi)
        self.assertEqual(store.keyword_search("gochugaru", 5), [])


class HybridRetrieveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = MemoryStore()
        cls.provider = FakeProvider()
        ingest_vault(cls.store, cls.provider, FIXTURE_VAULT, log=quiet)

    def test_hybrid_finds_the_note(self):
        chunks = retrieve(self.store, self.provider, "gochugaru", min_similarity=0.1, mode="hybrid")
        self.assertEqual(names(chunks)[0], "Kimchi")
        self.assertIn("score", chunks[0])

    def test_inline_filter_limits_primary_hits(self):
        chunks = retrieve(self.store, self.provider, "#book tides in the halls", min_similarity=0.0,
                          mode="hybrid")
        primary = [c for c in chunks if not c["via_link"]]
        self.assertTrue(primary)
        self.assertEqual(names(primary)[0], "Piranesi")
        self.assertTrue(all("book" in self.store.files[c["file"]]["tags"] for c in primary))

    def test_backlinks_pull_in_linking_notes(self):
        query = "Lighthouse App offline tide swims"
        without = retrieve(self.store, self.provider, query, top_k=1, min_similarity=0.1)
        with_bl = retrieve(self.store, self.provider, query, top_k=1, min_similarity=0.1, backlinks=True)
        self.assertEqual(names(with_bl)[0], "Lighthouse App")
        added = {c["file"] for c in with_bl} - {c["file"] for c in without}
        self.assertTrue(added)
        self.assertTrue(all(c["via"] == "backlink" for c in with_bl if c["file"] in added))
        self.assertIn("Sea Swimming", names(with_bl))

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            retrieve(self.store, self.provider, "x", mode="magic")


class ReindexTest(unittest.TestCase):
    def test_switching_chunking_reindexes_once(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            vault = tmp / "vault"
            shutil.copytree(FIXTURE_VAULT, vault)
            store, provider = MemoryStore(), FakeProvider()
            first = ingest_vault(store, provider, vault, log=quiet, chunking="window")
            again = ingest_vault(store, provider, vault, log=quiet, chunking="window")
            switched = ingest_vault(store, provider, vault, log=quiet, chunking="heading")
            self.assertEqual(again["changed"], 0)
            self.assertEqual(switched["changed"], first["changed"])
            note = vault / "Kitchen" / "Kimchi.md"
            self.assertEqual(index_hash(note, "heading").split(":")[1], "heading")
        finally:
            shutil.rmtree(tmp)

    def test_tags_are_stored(self):
        store = MemoryStore()
        ingest_vault(store, FakeProvider(), FIXTURE_VAULT, log=quiet)
        loaf = next(p for p in store.file_paths() if p.endswith("Weekend Loaf.md"))
        self.assertEqual(store.files[loaf]["tags"], ["baking", "recipe"])


class PostgresSqlTest(unittest.TestCase):
    """Only the pieces that don't need a database."""

    def test_tsquery_has_no_operators_from_the_user(self):
        self.assertEqual(_tsquery_words("what's the 'water_temp_c' & | ! <-> column?"),
                         ["water", "temp", "column"])
        self.assertEqual(_tsquery_words("the and of"), [])

    def test_filter_sql_is_parameterised(self):
        sql, params = PostgresStore._filter_sql({"tags": ["#Book"], "folder": "/Reading/"})
        self.assertNotIn("Book", sql)
        self.assertNotIn("Reading", sql)
        self.assertEqual(params, [["book"], "/reading/"])
        self.assertEqual(PostgresStore._filter_sql(None), ("", []))
        sql, params = PostgresStore._filter_sql({"folder": "Kitchen", "root": "/v/Vault"})
        self.assertIn("LEFT(LOWER(file_path), %s) = %s", sql)
        self.assertEqual(params, [len("/v/vault/kitchen/"), "/v/vault/kitchen/"])
        self.assertNotIn("LIKE", PostgresStore._filter_sql({"tags": ["to_do"]})[0])

    def test_keyword_terms(self):
        self.assertEqual(keyword_terms("Where is the NB04 station?"), ["nb04", "station"])


if __name__ == "__main__":
    unittest.main()
