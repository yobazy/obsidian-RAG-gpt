# web-and-local: progress notes

Branch `web-and-local` from `main`. Overnight run, 2026-10-28, on `pc`.

## Plan

1. Local backend: provider switch, Ollama embeddings + answers, per-provider tables.
2. API: `/ask` (streamed), `/search`, `/notes/{path}`, `/status`.
3. Frontend: Vite + React + TS, dark editorial look, screenshots.
4. Eval with the local provider on the fixture vault, README, self-review, draft PR.

## Environment limits hit tonight

The session's permission mode held these for approval, and nobody was there to approve:

- `pip install` (into the repo venv), so FastAPI, psycopg2, pytest, httpx are not installed.
- `docker run` / `docker compose up`, so no Postgres + pgvector container could start.
- Shell commands with inline env vars (`FOO=1 cmd`).

`npm install` inside `web/` was allowed. Python 3.12 has only the standard library plus `rich`.

What I did about it (decisions):

- The Ollama client uses `urllib` (stdlib). It needed no new dependency anyway.
- Added `MemoryStore`, an in-memory index with the same interface as the Postgres store. Tests, the eval on the fixture vault and the web demo use it (`RAG_STORE=memory`). The Postgres code is written but **not run against a real database tonight**.
- The API server (stage 2) uses the stdlib `http.server` in place of FastAPI so it could be run and tested. See stage 2.

## Stage 1: local backend (done)

- `rag/` package: `config.py` (settings, provider switch), `notes.py` (frontmatter, wikilinks, cleaning, chunking), `providers.py` (Ollama and Voyage/Claude), `store.py` (Postgres and memory), `pipeline.py` (ingest, retrieve, answer).
- `ingest.py`, `chat.py`, `watch.py`, `eval.py` are thin CLIs on top, each with `--provider` and `--vault`.
- Switch: `RAG_PROVIDER=cloud|local` (default `cloud`, so nothing changes for the existing setup).
- Tables: cloud keeps `chunks` (1024-dim), local uses `chunks_nomic_768` (768-dim). Migrations only add: `CREATE TABLE IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS heading`, a new `ingest_log` table. `setup()` refuses to run if the table's vector size doesn't match the provider, instead of writing bad rows.
- New tables get an HNSW index (ivfflat trained on an empty table gives poor recall). The existing `chunks` ivfflat index is left alone.
- Cold start: 180 s read timeout, one retry on connection errors, and `ProviderUnavailable` with a plain message ("Can't reach Ollama at ... Check that pc is on and Tailscale is connected."). `ping()` reports whether the models are loaded, so the UI can warn that the first answer will be slow.
- qwen3: `think: false`, plus a stream filter that drops a `<think>` block if one appears anyway.
- nomic-embed-text: `search_query:` / `search_document:` prefixes, as the model expects.

Smaller changes made along the way:

- Chunks now record the heading they start under, so answers can cite a section. The word windows are the same 400/50 as before.
- Wikilink text is kept in the embedded text (it used to be deleted, so "see [[Weekend Loaf]]" lost the name). Links are normalised: `[[Folder/Note#Heading|alias]]` is stored as `Note`; image embeds are skipped. Before, `Note|alias` was stored and never matched a file.
- Linked-note expansion takes the first 2 chunks per linked note (cap 10 total) so one long note can't crowd out the others.
- Ingest skips dot folders (`.obsidian`, `.trash`) and clears index rows for notes deleted from the vault (only rows under this vault's path).
- Emptied notes are removed from the index instead of keeping stale chunks.
- `CLAUDE_MODEL` is configurable; default stays `claude-sonnet-4-6` as before.

Tests: `python3 -m unittest discover -s tests -t .` (fake provider + memory store), and `python3 -m tests.live` for the real Ollama on pc.

## Needs Bazil

(filled in at the end)
