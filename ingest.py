import os
import re
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import voyageai
import psycopg2
from psycopg2.extras import execute_values

load_dotenv()

VAULT_PATH = os.getenv("VAULT_PATH")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

voyage = voyageai.Client(api_key=VOYAGE_API_KEY)

def setup_db(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector(1024),
                UNIQUE(file_path, chunk_index)
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING ivfflat (embedding vector_cosine_ops);
        """)
        conn.commit()
    print("✅ DB ready")

def get_markdown_files(vault_path):
    return list(Path(vault_path).rglob("*.md"))

def file_hash(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

def chunk_text(text, chunk_size=400, overlap=50):
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def clean_markdown(text):
    """Strip markdown syntax for cleaner embeddings."""
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # code blocks
    text = re.sub(r'#+ ', '', text)                          # headings
    text = re.sub(r'\[\[.*?\]\]', '', text)                  # obsidian links
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)               # markdown links
    text = re.sub(r'[*_`>-]', '', text)                      # formatting
    return text.strip()

def ingest_file(conn, file_path):
    raw = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_markdown(raw)
    
    if len(cleaned.strip()) < 50:
        return 0  # skip near-empty files

    fhash = file_hash(file_path)
    chunks = chunk_text(cleaned)

    # Check if file already ingested with same hash
    with conn.cursor() as cur:
        cur.execute(
            "SELECT file_hash FROM chunks WHERE file_path = %s LIMIT 1",
            (str(file_path),)
        )
        row = cur.fetchone()
        if row and row[0] == fhash:
            return 0  # unchanged, skip

        # Delete old chunks for this file if hash changed
        cur.execute("DELETE FROM chunks WHERE file_path = %s", (str(file_path),))

    # Embed all chunks in one batch call
    result = voyage.embed(chunks, model="voyage-3", input_type="document")
    embeddings = result.embeddings

    rows = [
        (str(file_path), fhash, i, chunk, emb)
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO chunks (file_path, file_hash, chunk_index, content, embedding)
            VALUES %s
            ON CONFLICT (file_path, chunk_index) DO UPDATE
            SET content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                file_hash = EXCLUDED.file_hash
        """, rows)
    conn.commit()
    return len(chunks)

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    setup_db(conn)

    files = get_markdown_files(VAULT_PATH)
    print(f"📂 Found {len(files)} markdown files in vault")

    total_chunks = 0
    for i, f in enumerate(files):
        count = ingest_file(conn, f)
        if count:
            print(f"  [{i+1}/{len(files)}] {f.name} → {count} chunks")
        total_chunks += count

    print(f"\n✅ Done. {total_chunks} chunks ingested.")
    conn.close()

if __name__ == "__main__":
    main()