"""Chunk storage: Postgres + pgvector, or an in-memory store for tests and demos.

Migrations here only ever add (CREATE ... IF NOT EXISTS, ADD COLUMN IF NOT
EXISTS). Nothing is dropped, and each provider writes to its own table.
"""

import math
import re
import threading
from datetime import datetime, timezone

from .config import Settings
from .notes import Chunk, note_name

LINKED_CHUNKS_PER_NOTE = 2
LINKED_CHUNKS_LIMIT = 10


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

    def replace_file(self, path, fhash, chunks: list[Chunk], embeddings, links):
        from psycopg2.extras import execute_values

        rows = [
            (path, fhash, c.index, c.content, emb, links, c.heading)
            for c, emb in zip(chunks, embeddings)
        ]
        with self._lock:
            try:
                with self.conn.cursor() as cur:
                    cur.execute(f"DELETE FROM {self.table} WHERE file_path = %s", (path,))
                    execute_values(
                        cur,
                        f"""INSERT INTO {self.table}
                            (file_path, file_hash, chunk_index, content, embedding, backlinks, heading)
                            VALUES %s""",
                        rows,
                        template=f"(%s, %s, %s, %s, %s::vector({self.dim}), %s, %s)",
                    )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def update_links(self, path, links):
        self._run("UPDATE {t} SET backlinks = %s WHERE file_path = %s", (links, path), commit=True)

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

    def search(self, embedding, top_k, min_similarity):
        vec = "[" + ",".join(f"{x:.7g}" for x in embedding) + "]"
        rows = self._run("""
            SELECT file_path, content, backlinks, 1 - (embedding <=> %s::vector) AS similarity,
                   heading, chunk_index
            FROM {t}
            WHERE 1 - (embedding <=> %s::vector) > %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (vec, vec, min_similarity, vec, top_k), fetch=True)
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

    def setup(self):
        pass

    def close(self):
        pass

    def file_hash(self, path):
        f = self.files.get(path)
        return f["hash"] if f else None

    def file_paths(self):
        return set(self.files)

    def replace_file(self, path, fhash, chunks, embeddings, links):
        with self._lock:
            self.files[path] = {
                "hash": fhash,
                "links": list(links),
                "chunks": [(c, _unit(e)) for c, e in zip(chunks, embeddings)],
            }

    def update_links(self, path, links):
        if path in self.files:
            self.files[path]["links"] = list(links)

    def delete_file(self, path):
        with self._lock:
            self.files.pop(path, None)

    def _row(self, path, chunk, similarity):
        return {"file": path, "content": chunk.content, "links": list(self.files[path]["links"]),
                "similarity": similarity, "heading": chunk.heading, "chunk_index": chunk.index}

    def search(self, embedding, top_k, min_similarity):
        q = _unit(embedding)
        scored = []
        for path, f in list(self.files.items()):
            for chunk, vec in f["chunks"]:
                sim = sum(a * b for a, b in zip(q, vec))
                if sim > min_similarity:
                    scored.append((sim, path, chunk))
        scored.sort(key=lambda s: -s[0])
        return [self._row(p, c, s) for s, p, c in scored[:top_k]]

    def chunks_for_names(self, names, exclude_files=()):
        wanted = {_note_key(n) for n in names}
        out = []
        for path in sorted(self.files):
            if path in exclude_files or _note_key(note_name(path)) not in wanted:
                continue
            for chunk, _ in self.files[path]["chunks"][:LINKED_CHUNKS_PER_NOTE]:
                out.append(self._row(path, chunk, 0.0))
        return out[:LINKED_CHUNKS_LIMIT]

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
