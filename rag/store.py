"""Chunk storage: Postgres + pgvector, or an in-memory store for tests and demos.

Migrations here only ever add (CREATE ... IF NOT EXISTS, ADD COLUMN IF NOT
EXISTS). Nothing is dropped, and each provider writes to its own table.
"""

import math
import re
import threading
from datetime import datetime, timezone

from .config import Settings
from .notes import Chunk, matches_filters, note_name

LINKED_CHUNKS_PER_NOTE = 2
LINKED_CHUNKS_LIMIT = 10
BACKLINK_CHUNKS_LIMIT = 5

# BM25 parameters for the in-memory keyword index (the usual defaults).
BM25_K1 = 1.2
BM25_B = 0.75

STOPWORDS = frozenset("""
a about after again all am an and any are as at be because been before being both but by
can could did do does doing down during each few for from further had has have having he
her here hers him his how i if in into is it its itself just me more most my myself no nor
not now of off on once only or other our ours out over own same she should so some such
than that the their theirs them then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you
your yours
""".split())


def keyword_terms(text: str) -> list[str]:
    """Lowercase word tokens with stopwords dropped."""
    return [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in STOPWORDS and len(w) > 1]


def _stem(word: str) -> str:
    """Plurals to singular for the memory index (tides -> tide, loaves stay).
    Postgres's english stemmer goes further; this is enough for the tests
    and the fixture eval."""
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _tsquery(query: str) -> str:
    """An OR query for to_tsquery built only from [a-z0-9] words, so no
    tsquery syntax from the user ever reaches Postgres. Postgres's parser
    splits `water_temp_c` into words, so do the same here."""
    words = []
    for term in keyword_terms(query):
        words.extend(w for w in term.split("_") if len(w) > 1)
    return " | ".join(dict.fromkeys(words))


class DimensionMismatch(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc)


def _note_key(name: str) -> str:
    return name.lower()


class PostgresStore:
    def __init__(self, settings: Settings, conn=None):
        import psycopg2  # imported here so memory mode needs no driver

        self._psycopg2 = psycopg2
        spec = settings.spec()
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", spec.table):
            raise ValueError(f"Unsafe table name {spec.table!r}")
        self.table = spec.table
        self.dim = spec.dim
        self.provider = spec.name
        self.conn = conn or psycopg2.connect(**settings.db)
        self._lock = threading.Lock()  # one connection, shared across threads

    def close(self):
        self.conn.close()

    def _run(self, sql, params=(), fetch=None, commit=False):
        with self._lock:
            try:
                with self.conn.cursor() as cur:
                    cur.execute(sql.replace("{t}", self.table), params)
                    rows = cur.fetchall() if fetch else None
                if commit:
                    self.conn.commit()
                elif not self.conn.autocommit:
                    self.conn.rollback()  # end read transactions; avoids idle-in-transaction
                return rows
            except Exception:
                self.conn.rollback()
                raise

    def setup(self):
        with self._lock, self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("SELECT to_regclass(%s)", (self.table,))
            exists = cur.fetchone()[0] is not None
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id SERIAL PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({self.dim}),
                    backlinks TEXT[] DEFAULT '{{}}',
                    heading TEXT,
                    UNIQUE(file_path, chunk_index)
                );
            """)
            # Older `chunks` tables predate these columns.
            cur.execute(f"ALTER TABLE {self.table} ADD COLUMN IF NOT EXISTS backlinks TEXT[] DEFAULT '{{}}';")
            cur.execute(f"ALTER TABLE {self.table} ADD COLUMN IF NOT EXISTS heading TEXT;")
            cur.execute(f"ALTER TABLE {self.table} ADD COLUMN IF NOT EXISTS tags TEXT[] DEFAULT '{{}}';")
            # Full-text column for hybrid search: note name, section and text.
            # Generated, so existing rows get it without re-embedding.
            cur.execute(f"""
                ALTER TABLE {self.table} ADD COLUMN IF NOT EXISTS tsv tsvector
                GENERATED ALWAYS AS (to_tsvector('english',
                    regexp_replace(file_path, '^.*/|\\.md$', '', 'g') || ' ' ||
                    coalesce(heading, '') || ' ' || content)) STORED;
            """)
            cur.execute("""
                SELECT atttypmod FROM pg_attribute
                WHERE attrelid = %s::regclass AND attname = 'embedding'
            """, (self.table,))
            dim = cur.fetchone()[0]
            if dim != self.dim:
                self.conn.rollback()
                raise DimensionMismatch(
                    f"Table {self.table} stores {dim}-dim vectors but the {self.provider} "
                    f"provider makes {self.dim}-dim ones. Point CHUNKS_TABLE_"
                    f"{self.provider.upper()} at another table."
                )
            if not exists:
                # HNSW works on an empty table; ivfflat needs data to train on.
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.table}_embedding_hnsw
                    ON {self.table} USING hnsw (embedding vector_cosine_ops);
                """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table}_file_path_idx ON {self.table} (file_path);
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table}_tsv_idx ON {self.table} USING gin (tsv);
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ingest_log (
                    id SERIAL PRIMARY KEY,
                    provider TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    started_at TIMESTAMPTZ NOT NULL,
                    finished_at TIMESTAMPTZ NOT NULL,
                    files_seen INTEGER NOT NULL,
                    files_changed INTEGER NOT NULL,
                    chunks_written INTEGER NOT NULL,
                    files_removed INTEGER NOT NULL DEFAULT 0
                );
            """)
        self.conn.commit()

    def file_hash(self, path):
        rows = self._run("SELECT file_hash FROM {t} WHERE file_path = %s LIMIT 1", (path,), fetch=True)
        return rows[0][0] if rows else None

    def file_paths(self):
        return {r[0] for r in self._run("SELECT DISTINCT file_path FROM {t}", fetch=True)}

    def replace_file(self, path, fhash, chunks: list[Chunk], embeddings, links, tags=()):
        from psycopg2.extras import execute_values

        rows = [
            (path, fhash, c.index, c.content, emb, links, c.heading, list(tags))
            for c, emb in zip(chunks, embeddings)
        ]
        with self._lock:
            try:
                with self.conn.cursor() as cur:
                    cur.execute(f"DELETE FROM {self.table} WHERE file_path = %s", (path,))
                    execute_values(
                        cur,
                        f"""INSERT INTO {self.table}
                            (file_path, file_hash, chunk_index, content, embedding, backlinks,
                             heading, tags)
                            VALUES %s""",
                        rows,
                        template=f"(%s, %s, %s, %s, %s::vector({self.dim}), %s, %s, %s::text[])",
                    )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def update_links(self, path, links, tags=None):
        if tags is None:
            self._run("UPDATE {t} SET backlinks = %s WHERE file_path = %s", (links, path), commit=True)
        else:
            self._run("UPDATE {t} SET backlinks = %s, tags = %s::text[] WHERE file_path = %s",
                      (links, list(tags), path), commit=True)

    def delete_file(self, path):
        self._run("DELETE FROM {t} WHERE file_path = %s", (path,), commit=True)

    def _rows(self, rows):
        return [
            {
                "file": r[0],
                "content": r[1],
                "links": list(r[2] or []),
                "similarity": float(r[3]),
                "heading": r[4],
                "chunk_index": r[5],
            }
            for r in rows
        ]

    @staticmethod
    def _filter_sql(filters):
        """SQL and params for {"tags": [...], "folder": "..."}, as in matches_filters."""
        sql, params = "", []
        tags = [t.lstrip("#").lower() for t in (filters or {}).get("tags") or []]
        if tags:
            sql += """ AND EXISTS (SELECT 1 FROM UNNEST(tags) AS have, UNNEST(%s::text[]) AS want
                                WHERE LOWER(have) = want OR LOWER(have) LIKE want || '/%%')"""
            params.append(tags)
        folder = ((filters or {}).get("folder") or "").strip("/")
        if folder:
            sql += " AND STRPOS(LOWER(file_path), %s) > 0"
            params.append(f"/{folder.lower()}/")
        return sql, params

    @staticmethod
    def _vec(embedding):
        return "[" + ",".join(f"{x:.7g}" for x in embedding) + "]"

    def search(self, embedding, top_k, min_similarity, filters=None):
        vec = self._vec(embedding)
        where, params = self._filter_sql(filters)
        rows = self._run(f"""
            SELECT file_path, content, backlinks, 1 - (embedding <=> %s::vector) AS similarity,
                   heading, chunk_index
            FROM {{t}}
            WHERE 1 - (embedding <=> %s::vector) > %s{where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (vec, vec, min_similarity, *params, vec, top_k), fetch=True)
        return self._rows(rows)

    def keyword_search(self, query, top_k, filters=None, embedding=None):
        """Full-text match on any query word, ranked by ts_rank with length
        normalisation (BM25-like, not exact BM25). `similarity` is the cosine
        to `embedding` when given, so callers can show it."""
        tsq = _tsquery(query)
        if not tsq:
            return []
        where, params = self._filter_sql(filters)
        if embedding is not None:
            sim, sim_params = "1 - (embedding <=> %s::vector)", [self._vec(embedding)]
        else:
            sim, sim_params = "0.0", []
        rows = self._run(f"""
            SELECT file_path, content, backlinks, {sim} AS similarity, heading, chunk_index
            FROM {{t}}, to_tsquery('english', %s) AS q
            WHERE tsv @@ q{where}
            ORDER BY ts_rank(tsv, q, 1) DESC, file_path, chunk_index
            LIMIT %s
        """, (*sim_params, tsq, *params, top_k), fetch=True)
        return self._rows(rows)

    def chunks_for_names(self, names, exclude_files=()):
        if not names:
            return []
        rows = self._run(f"""
            SELECT file_path, content, backlinks, 0.0, heading, chunk_index
            FROM {{t}}
            WHERE LOWER(REGEXP_REPLACE(SPLIT_PART(file_path, '/', -1), '\\.md$', '', 'i')) = ANY(%s)
              AND NOT (file_path = ANY(%s))
              AND chunk_index < {LINKED_CHUNKS_PER_NOTE}
            ORDER BY file_path, chunk_index
            LIMIT {LINKED_CHUNKS_LIMIT}
        """, ([_note_key(n) for n in names], list(exclude_files)), fetch=True)
        return self._rows(rows)

    def backlink_chunks(self, names, exclude_files=(), limit=BACKLINK_CHUNKS_LIMIT):
        """One chunk from each note that links to any of `names`: the first
        chunk whose text mentions the name, else the note's first chunk."""
        if not names:
            return []
        keys = [_note_key(n) for n in names]
        rows = self._run(f"""
            SELECT * FROM (
                SELECT DISTINCT ON (file_path) file_path, content, backlinks, 0.0, heading, chunk_index
                FROM {{t}}
                WHERE EXISTS (SELECT 1 FROM UNNEST(backlinks) AS x WHERE LOWER(x) = ANY(%s))
                  AND NOT (file_path = ANY(%s))
                ORDER BY file_path,
                         EXISTS (SELECT 1 FROM UNNEST(%s::text[]) AS n
                                 WHERE STRPOS(LOWER(content), n) > 0) DESC,
                         chunk_index
            ) picked
            ORDER BY file_path
            LIMIT {int(limit)}
        """, (keys, list(exclude_files), keys), fetch=True)
        return self._rows(rows)

    def links_of(self, path):
        rows = self._run("SELECT backlinks FROM {t} WHERE file_path = %s LIMIT 1", (path,), fetch=True)
        return list(rows[0][0] or []) if rows else []

    def backlinks_to(self, name):
        rows = self._run("""
            SELECT DISTINCT file_path FROM {t}
            WHERE %s = ANY(SELECT LOWER(x) FROM UNNEST(backlinks) AS x)
            ORDER BY file_path
        """, (_note_key(name),), fetch=True)
        return [r[0] for r in rows]

    def stats(self):
        rows = self._run("SELECT COUNT(*), COUNT(DISTINCT file_path) FROM {t}", fetch=True)
        return {"chunks": rows[0][0], "notes": rows[0][1]}

    def log_ingest(self, started, files_seen, files_changed, chunks_written, files_removed):
        self._run("""
            INSERT INTO ingest_log (provider, table_name, started_at, finished_at,
                                    files_seen, files_changed, chunks_written, files_removed)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (self.provider, self.table, started, _now(), files_seen, files_changed,
              chunks_written, files_removed), commit=True)

    def last_ingest(self):
        rows = self._run("""
            SELECT finished_at, files_seen, files_changed, chunks_written, files_removed
            FROM ingest_log WHERE table_name = %s ORDER BY finished_at DESC LIMIT 1
        """, (self.table,), fetch=True)
        if not rows:
            return None
        r = rows[0]
        return {"finished_at": r[0].isoformat(), "files_seen": r[1], "files_changed": r[2],
                "chunks_written": r[3], "files_removed": r[4]}


class MemoryStore:
    """Same interface, kept in a dict. Used by tests and `RAG_STORE=memory`."""

    def __init__(self, settings: Settings | None = None):
        spec = settings.spec() if settings else None
        self.provider = spec.name if spec else "test"
        self.table = "memory"
        self.dim = spec.dim if spec else None
        self.files: dict[str, dict] = {}
        self.log: list[dict] = []
        self._lock = threading.Lock()
        self._bm25 = None  # keyword index, rebuilt on the first search after a change

    def setup(self):
        pass

    def close(self):
        pass

    def file_hash(self, path):
        f = self.files.get(path)
        return f["hash"] if f else None

    def file_paths(self):
        return set(self.files)

    def replace_file(self, path, fhash, chunks, embeddings, links, tags=()):
        with self._lock:
            self.files[path] = {
                "hash": fhash,
                "links": list(links),
                "tags": list(tags),
                "chunks": [(c, _unit(e)) for c, e in zip(chunks, embeddings)],
            }
            self._bm25 = None

    def update_links(self, path, links, tags=None):
        if path in self.files:
            self.files[path]["links"] = list(links)
            if tags is not None:
                self.files[path]["tags"] = list(tags)

    def delete_file(self, path):
        with self._lock:
            self.files.pop(path, None)
            self._bm25 = None

    def _row(self, path, chunk, similarity):
        return {"file": path, "content": chunk.content, "links": list(self.files[path]["links"]),
                "similarity": similarity, "heading": chunk.heading, "chunk_index": chunk.index}

    def _passes(self, path, filters):
        return matches_filters(path, self.files[path].get("tags", []), filters)

    def search(self, embedding, top_k, min_similarity, filters=None):
        q = _unit(embedding)
        scored = []
        for path, f in list(self.files.items()):
            if not self._passes(path, filters):
                continue
            for chunk, vec in f["chunks"]:
                sim = sum(a * b for a, b in zip(q, vec))
                if sim > min_similarity:
                    scored.append((sim, path, chunk))
        scored.sort(key=lambda s: -s[0])
        return [self._row(p, c, s) for s, p, c in scored[:top_k]]

    def _keyword_index(self):
        with self._lock:
            if self._bm25 is None:
                docs = []
                for path, f in self.files.items():
                    for chunk, vec in f["chunks"]:
                        text = f"{note_name(path)} {chunk.heading or ''} {chunk.content}"
                        terms = [_stem(t) for t in keyword_terms(text)]
                        counts: dict[str, int] = {}
                        for t in terms:
                            counts[t] = counts.get(t, 0) + 1
                        docs.append((path, chunk, vec, counts, len(terms)))
                df: dict[str, int] = {}
                for doc in docs:
                    for t in doc[3]:
                        df[t] = df.get(t, 0) + 1
                avg = sum(doc[4] for doc in docs) / len(docs) if docs else 0.0
                self._bm25 = (docs, df, avg)
            return self._bm25

    def keyword_search(self, query, top_k, filters=None, embedding=None):
        """BM25 over note name, heading and chunk text."""
        terms = list(dict.fromkeys(_stem(t) for t in keyword_terms(query)))
        if not terms:
            return []
        docs, df, avg = self._keyword_index()
        n = len(docs)
        q = _unit(embedding) if embedding is not None else None
        scored = []
        for path, chunk, vec, counts, length in docs:
            if path not in self.files or not self._passes(path, filters):
                continue
            score = 0.0
            for t in terms:
                tf = counts.get(t)
                if not tf:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * length / avg))
            if score > 0:
                sim = sum(a * b for a, b in zip(q, vec)) if q is not None else 0.0
                scored.append((score, path, chunk, sim))
        scored.sort(key=lambda s: (-s[0], s[1], s[2].index))
        return [{**self._row(p, c, sim), "keyword_score": score} for score, p, c, sim in scored[:top_k]]

    def chunks_for_names(self, names, exclude_files=()):
        wanted = {_note_key(n) for n in names}
        out = []
        for path in sorted(self.files):
            if path in exclude_files or _note_key(note_name(path)) not in wanted:
                continue
            for chunk, _ in self.files[path]["chunks"][:LINKED_CHUNKS_PER_NOTE]:
                out.append(self._row(path, chunk, 0.0))
        return out[:LINKED_CHUNKS_LIMIT]

    def backlink_chunks(self, names, exclude_files=(), limit=BACKLINK_CHUNKS_LIMIT):
        keys = [_note_key(n) for n in names]
        out = []
        for path in sorted(self.files):
            f = self.files[path]
            if path in exclude_files or not f["chunks"]:
                continue
            if not {_note_key(x) for x in f["links"]} & set(keys):
                continue
            chunk = next((c for c, _ in f["chunks"] if any(k in c.content.lower() for k in keys)),
                         f["chunks"][0][0])
            out.append(self._row(path, chunk, 0.0))
        return out[:limit]

    def links_of(self, path):
        f = self.files.get(path)
        return list(f["links"]) if f else []

    def backlinks_to(self, name):
        key = _note_key(name)
        return sorted(p for p, f in self.files.items() if key in {_note_key(x) for x in f["links"]})

    def stats(self):
        return {"chunks": sum(len(f["chunks"]) for f in self.files.values()), "notes": len(self.files)}

    def log_ingest(self, started, files_seen, files_changed, chunks_written, files_removed):
        self.log.append({"finished_at": _now().isoformat(), "files_seen": files_seen,
                         "files_changed": files_changed, "chunks_written": chunks_written,
                         "files_removed": files_removed})

    def last_ingest(self):
        return self.log[-1] if self.log else None


def _unit(vec):
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def make_store(settings: Settings):
    if settings.store == "memory":
        return MemoryStore(settings)
    return PostgresStore(settings)
