# obsidian-rag-gpt

> Ask questions about your own notes. Get answers backed by what you actually wrote.

A local RAG (Retrieval-Augmented Generation) pipeline over your Obsidian vault. Ingests your markdown notes into pgvector, retrieves semantically relevant chunks via Voyage AI embeddings, and answers questions grounded in your actual notes using the Claude API.

---

## How it works

```
Obsidian Vault (.md files)
        ↓
Ingestion (ingest.py)          Auto re-ingestion (watch.py)
  — walks vault, chunks notes    — watches vault for saves
  — embeds via Voyage AI         — re-ingests changed notes on the fly
  — stores chunks + embeddings in pgvector
        ↓
pgvector (Postgres via Docker)
        ↓
Retrieval + Generation (chat.py)
  — semantic search over embeddings (with similarity threshold)
  — expands context via Obsidian [[backlinks]]
  — Claude answers grounded in retrieved context
        ↓
CLI chat interface
```

## Stack

- **Python** — ingestion, watching, and retrieval scripts
- **PostgreSQL + pgvector** — vector storage and similarity search (Docker)
- **Voyage AI** (`voyage-3`) — document and query embeddings
- **Claude API** (`claude-sonnet-4-6`) — answer generation
- **Watchdog** — filesystem watcher for auto re-ingestion
- **Rich** — CLI interface

---

## Getting started

### Prerequisites

- Python 3.10+
- Docker
- Obsidian vault with markdown notes
- [Voyage AI API key](https://dashboard.voyageai.com)
- [Anthropic API key](https://console.anthropic.com)

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/recall.git
cd recall
```

### 2. Set up Python environment

```bash
python -m venv venv
source venv/bin/activate
pip install anthropic voyageai psycopg2-binary pgvector watchdog rich python-dotenv
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Fill in your API keys and vault path in `.env`.

### 4. Start Postgres with pgvector

```bash
docker run -d \
  --name recall-db \
  -e POSTGRES_USER=your_db_user \
  -e POSTGRES_PASSWORD=your_db_password \
  -e POSTGRES_DB=obsidian_rag \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

Then enable the extension:

```bash
docker exec -it recall-db psql -U your_db_user -d obsidian_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 5. Ingest your vault

```bash
python ingest.py
```

This walks your vault, chunks each note, embeds via Voyage AI, and stores in pgvector. Re-ingestion is hash-based — unchanged files are skipped automatically.

### 6. (Optional) Watch for changes

Run this in a separate terminal to keep your index in sync as you edit notes:

```bash
python watch.py
```

It watches `VAULT_PATH` recursively and re-ingests any `.md` file the moment you save it. Press `Ctrl+C` to stop.

### 7. Start chatting

```bash
python chat.py
```

### 8. Run the eval harness

Score retrieval and answer quality against hand-written Q&A pairs from your notes:

```bash
# Full eval (retrieval + Claude answers)
python eval.py

# Retrieval only — fast, no LLM API calls
python eval.py --retrieval-only
```

Questions live in `eval/questions.json`. Each case lists `expected_keywords` that should appear in retrieved context and/or the generated answer.

---

## Features

- **Hash-based change detection** — only re-ingests notes that have changed
- **Auto re-ingestion** — `watch.py` keeps the index up to date as you edit notes in Obsidian
- **Backlink-aware retrieval** — expands retrieved context to include notes linked via `[[wikilinks]]`
- **Similarity threshold** — filters out low-confidence chunks before they reach the LLM
- **Source attribution** — every answer shows which notes it drew from, with similarity scores
- **Overlapping chunks** — 400-word chunks with 50-word overlap for better context continuity
- **Obsidian markdown stripping** — cleans `[[links]]`, headings, and formatting before embedding for cleaner vectors
- **Conversational memory** — chat maintains context across turns within a session
- **Eval harness** — scored Q&A pairs to measure retrieval and answer quality (`eval.py`)

## Roadmap

- [x] Watchdog-based auto re-ingestion on file save
- [x] Backlink-aware retrieval using Obsidian's `[[note]]` graph
- [x] Similarity threshold filtering
- [x] Eval harness with scored Q&A pairs
- [ ] React frontend

---

## License

MIT