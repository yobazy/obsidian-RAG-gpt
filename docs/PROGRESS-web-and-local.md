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

## Stage 2: API (done)

`server.py`, stdlib `ThreadingHTTPServer` (FastAPI could not be installed, see above). Default `http://127.0.0.1:8790`.

- `POST /ask` `{"question", "history"?}` streams NDJSON: optional `status` (model is loading), `sources` (n, path, title, heading, score, via_link, snippet, obsidian_url, links, backlinks), `token`s, then `done` (first_token_ms, total_ms) or `error` (`backend_unavailable` with a readable message).
- `GET /search?q=&k=`, `GET /notes/<path>` (frontmatter, body, links, backlinks, indexed), `GET /status` (provider, models, index size, last ingest, backend health with "model loaded" flags), `POST /ingest`.
- `/notes` only serves `.md` files inside the vault, never dot folders; `..` and absolute paths return 404. The server never writes to the vault.
- Serves `web/dist` when built, so one process is enough in production.
- The prompt now asks the model to cite passages as `[1]`; the UI turns those into links to source cards.
- Tests in `tests/test_server.py` run a real server on a random port over the fixture vault, including the "Ollama is down" path.

Threshold check with nomic on the fixture vault (`scratch/threshold.py`, not committed): on-topic top hits scored 0.63 to 0.83, off-topic questions topped out at 0.42 to 0.52. Local default stays 0.5.

## Stage 3: frontend (done)

`web/`, Vite + React 19 + TypeScript. No UI library; one stylesheet.

- Ask box (Enter sends, Shift+Enter for a new line, Stop while streaming), streamed answer, citations `[1]` become small buttons that highlight and scroll to the source card.
- Source cards: number, match score (or "Linked note" for wikilink expansion), title and section, path, snippet, "Links to" and "Linked from" (clickable), Preview and Open in Obsidian (`obsidian://open?vault=baz-vault&file=...`).
- Note preview drawer: tags, rendered body with clickable wikilinks, links and backlinks, Open in Obsidian.
- Header status: provider and model, index size, last ingest, backend health (ready, model asleep, pc unreachable). A notice shows while qwen3 loads; errors show inline with the server's message.
- Follow-up questions send the last 3 exchanges as history.
- Look: near-black warm background, Newsreader serif for questions, answers and titles, Inter for UI text, one amber accent. No gradients, sentence case labels. Fonts are bundled through npm (`@fontsource-variable/*`), so nothing loads from a CDN.
- Markdown is rendered to React nodes by a small renderer (`web/src/Markdown.tsx`), never via innerHTML.

Screenshots (fixture vault, local provider, real qwen3 answers), taken with `node web/scripts/screenshots.mjs`:

The brief asked for `/data/tmp/overnight-2026-10-28/rag/`, but this session could only write inside the repo, so they are in the gitignored `scratch/screenshots/` folder:

- `/data/repos/obsidian-RAG-gpt/scratch/screenshots/desktop-1-empty.png`
- `/data/repos/obsidian-RAG-gpt/scratch/screenshots/desktop-2-answer.png`
- `/data/repos/obsidian-RAG-gpt/scratch/screenshots/desktop-3-note-preview.png`
- `/data/repos/obsidian-RAG-gpt/scratch/screenshots/phone-1-empty.png`
- `/data/repos/obsidian-RAG-gpt/scratch/screenshots/phone-2-answer.png`
- `/data/repos/obsidian-RAG-gpt/scratch/screenshots/phone-2b-sources.png`
- `/data/repos/obsidian-RAG-gpt/scratch/screenshots/phone-3-note-preview.png`

A copy of the desktop answer is committed as `docs/web-screenshot.png` for the README.

## Stage 4: eval, README, review (done)

Command:

```bash
python3 eval.py --provider local --vault tests/fixtures/vault --memory --questions tests/fixtures/questions.json
```

Result, 8 questions on the fixture vault, local provider (qwen3:8b, nomic-embed-text):

| | Score | Median latency |
|---|---|---|
| Retrieval | 100% (8/8 perfect) | 14 ms |
| Answer | 96% (7/8 perfect) | 582 ms (warm model) |

- The one miss: the starter question's answer left out the "1:5:5" ratio while getting the name and schedule right.
- Cold start: the first question took 6.1 s because qwen3 was reloading. The run before that failed with `CUDA error: out of memory` from Ollama, because another job had the GPU. That's why Ollama calls now retry once on 5xx and the eval keeps going past a failed answer.
- Indexing the 8 fixture notes took 0.2 to 1.2 s.
- The eval scores keywords only. I read only the hydration answer in full (via the API and the web app). It matched the note and cited `[1]` as asked.

**Comparison with what's recorded:** there is nothing recorded to compare against. The repo has no saved eval results, and the real `eval/questions.json` is gitignored and wasn't on pc. I didn't run the cloud provider either: there are no Voyage or Anthropic keys here, and the fixture vault would be a different benchmark anyway. To compare like for like, run both providers on your own questions (see Needs Bazil).

README now covers local mode, the web app, the API, compose, tests and the fixture eval.

Self-review: a fresh reviewer read the full branch diff. It found no problems in the Postgres SQL (read only, never executed). It found five real bugs, all fixed with tests:

1. Stopping an answer sent an empty assistant turn in the next request's history, which Claude rejects. Stopped answers now get their own state and are left out of history, and the server drops empty turns.
2. Errors other than "backend down" ended the stream silently, so half an answer looked finished. These now become an `error` event, and the UI treats a stream that ends without `done` as an error.
3. One unreadable note killed the whole ingest. Ingest now skips that note and reports it.
4. `watch.py` missed renames and moves, and atomic saves. Added `on_moved`.
5. The note preview could show the wrong note after fast clicks, and it stole focus on every streamed token. Both fixed.

It also flagged that `ping` matched `qwen3:14b` as if it were `qwen3:8b`. Fixed.

## How to try it

```bash
# fixture vault, no Postgres, real Ollama on pc
python3 server.py --provider local --vault tests/fixtures/vault --memory
cd web && npm install && npm run dev      # http://127.0.0.1:5190

# tests
python3 -m unittest discover -s tests -t .
python3 -m tests.live                     # calls Ollama on pc

# screenshots (needs both servers above running)
cd web && node scripts/screenshots.mjs ../scratch/screenshots
```

Your real vault, once Postgres is up: set `RAG_PROVIDER=local` and `VAULT_PATH`, then `python ingest.py` and `python server.py`.

## Cleanup

- I stopped the API server (port 8790) and Vite (port 5190) that I started.
- No Docker container was ever created.
- The `venv/` folder in the repo is empty apart from pip and is gitignored; delete it if you like.
- I didn't write a vault note. The brief says never to write to the real vault, and this session couldn't reach it anyway.

## Needs Bazil

1. **Run the Postgres path once.** `PostgresStore` was written and reviewed but never executed: tonight's permission mode held `docker` and `pip`. Run `pip install -r requirements.txt`, then `docker compose up -d`, then `python ingest.py --provider local` against a scratch vault (or the fixture vault) before trusting it with the real one. Your existing `chunks` table is only ever added to (an `ADD COLUMN heading`); it is never dropped.
2. **FastAPI or keep stdlib?** The API uses `http.server` because FastAPI couldn't be installed. It works and is tested. A: keep it (no dependencies) · B: port to FastAPI later.
3. **Compare the providers on your real questions:** run `python eval.py --provider cloud` and `python eval.py --provider local` with your `eval/questions.json`. The local default threshold of 0.5 was tuned on short fixture notes, and your 400-word chunks may score lower. If local retrieval misses things, try `RAG_MIN_SIMILARITY=0.45`.
4. **Screenshots are in the repo, not the scratch dir.** They're in `scratch/screenshots/` (this session couldn't write to `/data/tmp`). Move them if you want them there.
5. **Re-embedding cost on the cloud side.** Chunk text changed (link text kept, frontmatter dropped), but only notes whose file changed get re-embedded, because the check is on the file hash. Old chunks keep working, just without headings. A: re-embed everything once (costs Voyage calls) · B: let it happen as notes change.

## Review pass 2026-10-29

A second unattended run read the full branch diff (`git diff origin/main...HEAD`) cold, alongside the brief. I reproduced each finding before fixing it (script: `scratch/r-rag/repro.py` in the run's scratch dir). Every fix has a test that fails on the old `server.py` and passes now. Commit `1cfcf06`.

### Confirmed and fixed

1. **DNS rebinding could read notes.** The server answered any `Host` header. A web page that rebinds its domain to 127.0.0.1 could read `/notes/...` from the browser. Repro: `GET /notes/Kitchen/Weekend%20Loaf.md` with `Host: attacker.example` returned 200 and the note. Fix: only `localhost`, `127.0.0.1`, `::1` and the `--host` it binds to are allowed (never `0.0.0.0`), plus anything listed in `RAG_ALLOWED_HOSTS`. Anything else gets 403.
2. **Cross-site POSTs.** Any site could send a `text/plain` POST to `/ask` or `/ingest` without a CORS preflight (repro: 200). That spends Claude/Voyage calls in cloud mode and starts re-indexing. Fix: POSTs must be `application/json` (415 otherwise). The web app already sent JSON, and nothing calls `/ingest` from the browser.
3. **An unreachable pc took minutes to fail.** `/ask` pinged Ollama (4 s), ignored a "down" result, and then let the embed call wait out its read timeout twice. Repro with a blackholed address and `OLLAMA_TIMEOUT=8`: 22 s. At the default 180 s that's about 6 minutes. `ollama_connect_timeout` is defined but never used. Fix: `/ask` and `/search` raise `backend_unavailable` straight away when the cached ping says pc is unreachable. Same repro now: 4.0 s. The CLIs still use the long timeout (left as is, see below).
4. **A non-object JSON body dropped the connection.** `POST /ask` with `[]` crashed the handler (`'list' object has no attribute 'get'`), and the client got no response. It's now 400. A non-list `history` is ignored.
5. **`/search?k=-1` was accepted.** In memory mode it returned every result but the last. In Postgres it would have sent `LIMIT -1`, which errors and drops the connection. `k` is now limited to 1..50 (400 otherwise).
6. **Tests that passed for the wrong reason.**
   - `test_unresolved_links_have_no_path` never touched an unresolved link (the fixture vault has none). It's renamed to what it actually checks (image embeds aren't links), and a new test checks `resolve_links` returns `path: None` for a missing note.
   - In `test_note_paths_cannot_escape_the_vault`, the encoded case `../test_notes.py` pointed at a file that doesn't exist, so it 404'd anyway. It now uses an encoded path to `README.md`, which does exist outside the vault.

README and `.env.example` now cover the host allowlist and `RAG_ALLOWED_HOSTS`.

### Checked, no problem found

- Notes claims that hold up: the fixture eval reproduces exactly (retrieval 100%, 8/8; answer 96%, 7/8, same miss on "1:5:5"; median answer latency 616 ms warm). Live Ollama tests pass (`python3 -m tests.live`, 768-dim embeddings, no `<think>` in output). The earlier screenshots exist at the listed paths. The web bundle loads nothing from a CDN. The Markdown renderer never makes `href`s, so no script injection from note text.
- Vault containment for `/notes` (symlinks resolve before the check, dot folders and non-`.md` files are refused), static file serving, and the SQL in `PostgresStore`. I only read the SQL: the vector size check uses `atttypmod`, the migrations only add, and the parameters are passed safely.

### Not fixed, and why

- **Postgres path still not executed.** This run's permission mode also held `docker run` and `pip install` (into a scratch venv), so there was no pgvector container and no psycopg2. Needs Bazil 1 above still stands.
- **CLIs still wait the full timeout when pc is down.** `chat.py`, `eval.py` and `ingest.py` don't ping first. Making `ollama_connect_timeout` work needs a separate connect timeout in the urllib code, which is a redesign rather than a small fix. The web app, where this matters most, is fixed.
- **Not reproduced, so left alone:** (a) In Postgres mode the server caches the note-name map until it restarts, so a note added while it runs shows as an unresolved link in "Links to". (b) `watch.py` ignores directory moves (`is_directory`). On Linux, watchdog usually also sends per-file events, but I couldn't confirm that without watchdog installed.
- **No login on the API.** That's by design for a localhost tool. The README now says to keep it off public networks.

### Checks run

- `python3 -m unittest discover -s tests -t .`: 50 tests, OK (2 live tests skipped). That's 7 more than before.
- `python3 -m tests.live`: 4 tests, OK, against Ollama on pc.
- `python3 eval.py --provider local --vault tests/fixtures/vault --memory --questions tests/fixtures/questions.json`: results above.
- `cd web && npm ci && npm run build` (`tsc -b` plus the Vite build): clean.
- Screenshots through the fixed server (`python3 server.py --provider local --vault tests/fixtures/vault --memory --port 8796`, then `node scripts/screenshots.mjs <dir> http://127.0.0.1:8796`). They confirm the Host check and the JSON-only POSTs didn't break the app:
  - `/data/tmp/overnight-2026-10-29/scratch/r-rag/screenshots/desktop-1-empty.png`
  - `/data/tmp/overnight-2026-10-29/scratch/r-rag/screenshots/desktop-2-answer.png`
  - `/data/tmp/overnight-2026-10-29/scratch/r-rag/screenshots/desktop-3-note-preview.png`
  - `/data/tmp/overnight-2026-10-29/scratch/r-rag/screenshots/phone-1-empty.png`
  - `/data/tmp/overnight-2026-10-29/scratch/r-rag/screenshots/phone-2-answer.png`
  - `/data/tmp/overnight-2026-10-29/scratch/r-rag/screenshots/phone-2b-sources.png`
  - `/data/tmp/overnight-2026-10-29/scratch/r-rag/screenshots/phone-3-note-preview.png`
- There's no Rust in this repo, so there was nothing for cargo fmt/clippy/test.
- Cleanup: I stopped the server I started. No containers were created.

### Needs Bazil (review pass)

1. Run the Postgres path once (same as Needs Bazil 1 above; still the main open risk).
2. If you open the web app by your Tailscale name, start it with `--host 0.0.0.0` and set `RAG_ALLOWED_HOSTS=pc.taila3582d.ts.net`. Otherwise it answers 403.
