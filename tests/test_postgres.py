"""PostgresStore against a real pgvector database.

Skipped unless RAG_TEST_PG_DSN is set, e.g.
  RAG_TEST_PG_DSN="host=127.0.0.1 port=5432 dbname=rag_test user=rag password=rag"
Use a scratch database: the tests create and drop tables named test_chunks_*.
"""

import os
import unittest

from rag.config import Settings
from rag.pipeline import ingest_vault, retrieve
from rag.store import MemoryStore, PostgresStore
from tests.fakes import DIM, FIXTURE_VAULT, FakeProvider

DSN = os.getenv("RAG_TEST_PG_DSN")
try:
    import psycopg2
except ImportError:
    psycopg2 = None


def quiet(*_):
    pass


def names(rows):
    return [r["file"].rsplit("/", 1)[-1].removesuffix(".md") for r in rows]


@unittest.skipUnless(DSN and psycopg2, "set RAG_TEST_PG_DSN (and install psycopg2) to run")
class PostgresCase(unittest.TestCase):
    table = None

    @classmethod
    def make_store(cls, table, dim=DIM):
        store = PostgresStore(Settings(provider="local"), conn=psycopg2.connect(DSN))
        store.table, store.dim = table, dim
        return store

    @staticmethod
    def drop(conn, *tables):
        with conn.cursor() as cur:
            for t in tables:
                assert t.startswith("test_chunks_")
                cur.execute(f"DROP TABLE IF EXISTS {t}")
            cur.execute("SELECT to_regclass('ingest_log')")
            if cur.fetchone()[0] is not None:
                cur.execute("DELETE FROM ingest_log WHERE table_name = ANY(%s)", (list(tables),))
        conn.commit()


class MigrationTest(PostgresCase):
    """setup() on tables that older code made and filled."""

    MAIN = "test_chunks_main"  # schema from main: no heading, ivfflat index
    WEB = "test_chunks_web"  # schema from web-and-local: heading, hnsw index

    def setUp(self):
        self.conn = psycopg2.connect(DSN)
        self.drop(self.conn, self.MAIN, self.WEB)
        rows = [("/v/Kitchen/Kimchi.md", "h1", 0, "Kimchi. Gochugaru from the Korean shop.", ["Pantry"]),
                ("/v/Kitchen/Pantry.md", "h2", 0, "Pantry staples: flour, salt, rice.", []),
                ("/v/Bikes/Moorland Loop.md", "h3", 0, "Coffee at The Kettle on the moor.", [])]
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""CREATE TABLE {self.MAIN} (
                id SERIAL PRIMARY KEY, file_path TEXT NOT NULL, file_hash TEXT NOT NULL,
                chunk_index INTEGER NOT NULL, content TEXT NOT NULL, embedding vector({DIM}),
                backlinks TEXT[] DEFAULT '{{}}', UNIQUE(file_path, chunk_index))""")
            cur.execute(f"""CREATE TABLE {self.WEB} (
                id SERIAL PRIMARY KEY, file_path TEXT NOT NULL, file_hash TEXT NOT NULL,
                chunk_index INTEGER NOT NULL, content TEXT NOT NULL, embedding vector({DIM}),
                backlinks TEXT[] DEFAULT '{{}}', heading TEXT, UNIQUE(file_path, chunk_index))""")
            # main built its ivfflat index on the empty table, before the first ingest.
            cur.execute(f"CREATE INDEX {self.MAIN}_embedding_idx ON {self.MAIN} "
                        "USING ivfflat (embedding vector_cosine_ops)")
            for table in (self.MAIN, self.WEB):
                for path, h, i, content, links in rows:
                    vec = PostgresStore._vec(FakeProvider._vector(content))
                    cur.execute(f"""INSERT INTO {table}
                        (file_path, file_hash, chunk_index, content, embedding, backlinks)
                        VALUES (%s, %s, %s, %s, %s::vector, %s)""", (path, h, i, content, vec, links))
            cur.execute(f"CREATE INDEX {self.WEB}_embedding_hnsw ON {self.WEB} "
                        "USING hnsw (embedding vector_cosine_ops)")
        self.conn.commit()

    def tearDown(self):
        self.drop(self.conn, self.MAIN, self.WEB)
        self.conn.close()

    def test_old_tables_get_tsv_tags_and_index_without_reembedding(self):
        for table in (self.MAIN, self.WEB):
            with self.subTest(table=table):
                store = self.make_store(table)
                store.setup()
                store.setup()  # running it again changes nothing
                with self.conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*), COUNT(tsv), COUNT(*) FILTER (WHERE tags = '{{}}'),"
                                f" COUNT(DISTINCT file_hash) FROM {table}")
                    self.assertEqual(cur.fetchone(), (3, 3, 3, 3))
                    cur.execute("SELECT indexname, indexdef FROM pg_indexes WHERE tablename = %s",
                                (table,))
                    indexes = dict(cur.fetchall())
                self.conn.commit()
                self.assertIn(f"{table}_tsv_idx", indexes)
                self.assertIn(f"{table}_embedding_hnsw", indexes)
                self.assertFalse([d for d in indexes.values() if "ivfflat" in d])
                # Make the planner use the vector index even on three rows.
                with store.conn.cursor() as cur:
                    cur.execute("SET enable_seqscan = off")
                store.conn.commit()
                self.assertEqual(names(store.keyword_search("gochugaru", 5)), ["Kimchi"])
                # The note name is indexed too, not only the text.
                self.assertEqual(names(store.keyword_search("moorland", 5)), ["Moorland Loop"])
                self.assertEqual(store.stats(), {"chunks": 3, "notes": 3})
                emb = FakeProvider._vector("where do I get gochugaru")
                self.assertEqual(len(store.search(emb, 5, -1.0)), 3)
                self.assertEqual(names(store.search(emb, 5, 0.1, {"folder": "Kitchen"}))[0], "Kimchi")
                store.close()


class StoreTest(PostgresCase):
    """The same fixture vault in Postgres and in memory should behave alike."""

    TABLE = "test_chunks_store"

    @classmethod
    def setUpClass(cls):
        cls.store = cls.make_store(cls.TABLE)
        cls.drop(cls.store.conn, cls.TABLE)
        cls.store.setup()
        cls.provider = FakeProvider()
        ingest_vault(cls.store, cls.provider, FIXTURE_VAULT, log=quiet)
        cls.memory = MemoryStore()
        ingest_vault(cls.memory, cls.provider, FIXTURE_VAULT, log=quiet)

    @classmethod
    def tearDownClass(cls):
        cls.drop(cls.store.conn, cls.TABLE)
        cls.store.close()

    def test_same_index_as_memory(self):
        self.assertEqual(self.store.stats(), self.memory.stats())
        self.assertEqual(self.store.file_paths(), self.memory.file_paths())
        again = ingest_vault(self.store, self.provider, FIXTURE_VAULT, log=quiet)
        self.assertEqual(again["changed"], 0)

    def test_tags_are_stored(self):
        loaf = next(p for p in self.store.file_paths() if p.endswith("Weekend Loaf.md"))
        rows = self.store._run("SELECT DISTINCT tags FROM {t} WHERE file_path = %s", (loaf,), fetch=True)
        self.assertEqual(rows, [(["baking", "recipe"],)])

    def test_keyword_search(self):
        self.assertEqual(names(self.store.keyword_search("where do I get gochugaru", 5))[0], "Kimchi")
        self.assertEqual(self.store.keyword_search("what is the", 5), [])
        self.assertEqual(self.store.keyword_search("the & | !", 5), [])
        rows = self.store.keyword_search("tide", 10, {"tags": ["book"]})
        self.assertEqual(set(names(rows)), {"Piranesi"})
        rows = self.store.keyword_search("tide", 10, {"folder": "Projects", "root": str(FIXTURE_VAULT)})
        self.assertTrue(rows and all("/Projects/" in r["file"] for r in rows))

    def test_rare_words_outweigh_common_ones(self):
        # The stemmer turns "weekly" into "week", which is in every section of
        # the six-week course note. Without IDF those sections fill the list
        # and push out the call notes, the only other note with "questions".
        rows = self.store.keyword_search("What questions do I ask in the weekly review?", 2)
        self.assertEqual(names(rows), ["Weekly Review", "2026-09-03 Harbour Authority Call"])
        self.assertEqual(names(rows), names(self.memory.keyword_search(
            "What questions do I ask in the weekly review?", 2)))

    def test_similarity_comes_with_keyword_hits(self):
        emb = self.provider.embed(["gochugaru"], kind="query")[0]
        row = self.store.keyword_search("gochugaru", 1, embedding=emb)[0]
        self.assertGreater(row["similarity"], 0)

    def test_filtered_vector_search(self):
        emb = self.provider.embed(["tides in the halls"], kind="query")[0]
        rows = self.store.search(emb, 5, 0.0, {"tags": ["book"]})
        self.assertTrue(rows)
        self.assertEqual(names(rows)[0], "Piranesi")

    def test_links_and_backlinks_match_memory(self):
        lighthouse = [p for p in self.store.file_paths() if p.endswith("Lighthouse App.md")]
        pg = self.store.backlink_chunks(["Lighthouse App"], lighthouse)
        mem = self.memory.backlink_chunks(["Lighthouse App"], lighthouse)
        self.assertTrue(pg)
        self.assertEqual([(r["file"], r["chunk_index"]) for r in pg],
                         [(r["file"], r["chunk_index"]) for r in mem])
        self.assertEqual(self.store.chunks_for_names(["Sourdough Starter", "Pantry Staples"]),
                         self.memory.chunks_for_names(["Sourdough Starter", "Pantry Staples"]))
        self.assertEqual(self.store.backlinks_to("Sourdough Starter"),
                         self.memory.backlinks_to("Sourdough Starter"))
        self.assertEqual(self.store.links_of(lighthouse[0]), self.memory.links_of(lighthouse[0]))

    def test_hybrid_retrieve_with_filters(self):
        chunks = retrieve(self.store, self.provider, "#book tides in the halls", min_similarity=0.0,
                          mode="hybrid", backlinks=True, vault_root=FIXTURE_VAULT)
        primary = [c for c in chunks if not c["via_link"]]
        self.assertEqual(names(primary)[0], "Piranesi")
        for c in primary:
            tags = self.store._run("SELECT tags FROM {t} WHERE file_path = %s LIMIT 1",
                                   (c["file"],), fetch=True)[0][0]
            self.assertIn("book", tags)

    def test_ingest_log(self):
        self.assertEqual(self.store.last_ingest()["files_seen"], self.memory.last_ingest()["files_seen"])


if __name__ == "__main__":
    unittest.main()
