# hybrid-retrieval: progress notes

Branch `hybrid-retrieval` (local `ov/f-rag-hybrid`) from `web-and-local`. Overnight run, 2026-10-29, on `pc`.

## Plan

1. Grow the fixture vault and the eval set to 40+ questions with expected source notes (multi-hop and backlink cases). Add source-level metrics to `eval.py`. Record a baseline per provider.
2. Hybrid retrieval: keyword search (Postgres full-text, BM25 in memory) plus vectors, merged with reciprocal rank fusion. Tag and folder filter. Heading-aware chunking.
3. Optional rerank with qwen3 on pc, off by default, measured.

## Environment limits hit tonight

Same as the earlier runs: the permission mode held `pip install` (even into a scratch venv) and `docker run`, so there is no psycopg2 and no pgvector container. Postgres SQL is written but not executed. Every measured number below comes from the in-memory store, which implements the same interface.

## Stage 1: eval set and baseline (done)

Fixture vault: 16 new made-up notes (now 25 files, 24 indexed). They add near-duplicates on purpose so retrieval has to choose: a second bread recipe (Focaccia, 80 percent vs the loaf's 75), a second ride (Moorland Loop, with a cafe called The Kettle next to Kettle Cove), two pressures (tyres, boiler), tides in a novel (Piranesi), people and meeting notes that only link to the topic note (for backlink questions), and folders for filter questions.

`tests/fixtures/questions.json`: 51 questions, each with `type`, `expected_keywords` and `expected_sources` (note names). Types: fact 26, keyword 5 (rare tokens like `water_temp_c`, `gochugaru`), paraphrase 4 (no words shared with the note), multi-hop 6 (expected sources list every note on the chain), backlink 5 (the answer lives in a note that links to the subject), filter 3 (with a `filters` field: tags or folder; vague on purpose, so only the filter makes them answerable), abstain 2 (nothing in the vault answers them).

`eval.py` now scores sources as well as keywords:

- **Source recall**: share of expected notes that reach the model (primary hits and linked notes).
- **@3**: the same, counting only the first three distinct notes.
- **MRR / hit@1**: rank of the first expected note.
- **Abstain**: share of abstain questions where nothing above the threshold came back.
- Per-type breakdown, and `--json-out` writes the summary too. Abstain cases are kept out of the keyword averages.

`tests/test_eval.py` checks every expected source is a real fixture note, the set has 40+ questions with 5+ multi-hop and 5+ backlink cases, and the scoring maths.

Recall is close to the ceiling on a vault this small (7 chunks per question cover a lot of 24 notes), so **MRR, hit@1 and @3 are the numbers to watch**.

### Baseline, local provider (nomic-embed-text + qwen3:8b), vector only

```bash
python3 eval.py --provider local --vault tests/fixtures/vault --memory --questions tests/fixtures/questions.json
```

| Metric | Baseline |
|---|---|
| Source recall | 97% |
| Recall @3 | 93% |
| MRR | 0.92 |
| Hit@1 | 88% |
| Keyword retrieval | 95% |
| Abstain | 0% (0/2) |
| Chunks per question | 7.1 |
| Answer keywords | 92% |
| Median retrieval latency | 14 ms |
| Median answer latency | 671 ms (warm) |

By type (recall / @3 / MRR): fact 100/100/0.98, keyword 100/100/1.00, paraphrase 100/100/1.00, multi-hop 100/94/1.00, backlink 88/77/0.70, filter 67/33/0.40.

A retrieval-only run gave MRR 0.90 and hit@1 84% before I fixed the expected sources of two multi-hop questions to include the note the chain starts from. Numbers above are after that fix.

**Cloud provider: not run.** There's no `.env`, no Voyage or Anthropic key, and the `anthropic`/`voyageai` packages aren't installed (pip is held). See Needs Bazil.

## Stage 2: hybrid retrieval, filters, heading-aware chunks (done)

### What changed

- **Keyword search.** `store.keyword_search()`:
  - Memory store: real BM25 (k1 1.2, b 0.75) over note name, heading and chunk text, with stopwords dropped and plurals folded. The index rebuilds lazily after a write.
  - Postgres: a generated `tsv` column (`to_tsvector('english', note name || heading || content)`) with a GIN index, queried with an OR `to_tsquery` built only from `[a-z0-9]` words (no user syntax reaches Postgres) and ranked by `ts_rank(..., 1)` (length-normalised). That's BM25-like, not exact BM25; real BM25 in Postgres needs an extension (`pg_search`/ParadeDB), which I didn't want to add unrun. Generated column means existing rows get it with no re-embedding.
- **Fusion.** `pipeline.fuse()`: reciprocal rank fusion, k = 60, 20 candidates from each ranker, matched on (file, chunk). `similarity` stays the cosine score (the web app's "% match"); the fused score is in `score`.
- **Keyword-only hits are gated.** A row found only by keywords is kept if its cosine is at most 0.05 below the threshold. Without this, a stray word match could put an unrelated note in the context.
- **Filters.** Frontmatter tags are stored per note (`tags TEXT[]` in Postgres, added with `ADD COLUMN IF NOT EXISTS`). Filters come from the question (`#tag`, `tag:x`, `in:Folder`, `in:"Two Words"`) or a `filters` argument. They apply to both rankers' primary hits; linked notes are still added. `#12` or `C#` are not tags. Folder match is on a `/Folder/` path segment, so a vault that itself sits under a folder of the same name would over-match (unlikely, noted).
- **Heading-aware chunking** (`RAG_CHUNKING=heading`): notes over 400 words split at every heading; sections under 25 words merge into the next; long sections still get 400/50 windows. Each chunk's embedded and keyword-indexed text starts with "Note > Section". Notes that fit in one chunk stay whole (see measurements). The stored hash gets a `:heading` suffix, so switching modes re-indexes each note once instead of mixing chunk styles.
- **Backlink expansion** (`RAG_BACKLINKS=1`): adds one chunk from each note that links to a top hit (the chunk that mentions it, else the first), up to 5. They're labelled for the model as "a note that links to one of the passages above" and in the web app as "Links here".
- Tag updates ride along with link updates when a note's hash hasn't changed, so tags appear on old rows without re-embedding.
- `eval.py`: `--retrieval`, `--chunking`, `--backlinks/--no-backlinks`, `--rerank/--no-rerank`, `--top-k`; questions' `filters` are passed through; `--json-out` records the config.
- Six more questions (`long-note`, 57 total) over two new long notes (a 7-day trip diary, a 6-week course), because none of the earlier notes were over 400 words, so the eval couldn't see what heading chunking is for.

### Measurements

Fixture vault, local provider, 57 questions. Retrieval-only runs except where it says answers.

| Config | Recall | @3 | MRR | Hit@1 | Long-note MRR | Backlink recall | Filter recall | Chunks/q |
|---|---|---|---|---|---|---|---|---|
| Stage 1 code (vector, window, no filters), full run | 95% | 92% | 0.90 | 85% | 0.75 | 88% | 67% | 7.0 |
| vector, window (filters now applied) | 95% | 94% | 0.92 | 87% | 0.75 | 88% | 67% | 6.7 |
| vector, heading | 97% | 95% | 0.95 | 91% | 1.00 | 88% | 67% | 6.4 |
| hybrid, window | 98% | 96% | 0.95 | 93% | 0.83 | 95% | 100% | 6.6 |
| hybrid, heading | 100% | 97% | 0.97 | 95% | 1.00 | 95% | 100% | 6.5 |
| **hybrid, heading, backlinks (new default)** | **100%** | **97%** | **0.97** | **95%** | **1.00** | **100%** | **100%** | 7.2 |

Full runs with answers (qwen3:8b, warm):

| | Stage 1 code | New defaults |
|---|---|---|
| Source recall / MRR / hit@1 | 95% / 0.90 / 85% | 100% / 0.97 / 95% |
| All expected sources found | 93% | 100% |
| Answer keywords | 90% | 97% |
| Median retrieval latency | 15 ms | 14 ms |
| Median answer latency | 741 ms | 698 ms |

Things I tried and dropped:

- **Splitting every note at headings**, short ones included. Long-note MRR went to 1.00, but backlink MRR fell from 0.70 to 0.45 and multi-hop @3 from 94% to 72%: three pieces of one short note filled the top five and pushed other notes out. Short notes now stay whole.
- **Putting the note title in front of whole short notes** too. Backlink MRR 0.50 against 0.70 without it. The title is only added to section chunks of long notes.
- **Fusion constants.** In-process sweep (`scratch/.../sweep.py`): RRF k 10 vs 60 and candidate depth 10/20/40 changed nothing on this vault. The keyword gate margin: 0 lost a filter question (recall 98%), 0.05 was best (hit@1 92.7%), 0.1 to 1.0 were one question worse (90.9%). Took 0.05. That's a one-question difference, so treat it as "a small margin", not a tuned value.
- **First stemmer** (stripping -ed/-ing/-es) turned "tides" into "tid" and left "tide" alone, so they never matched. A test caught it; it now only folds plurals.

Noise: Ollama embeddings aren't bit-for-bit stable between runs, so the same config can move by one question (fact MRR 0.98 in one run, 1.00 in another).

Abstain stayed at 0% in every config: the passport and Kubernetes questions both score above the 0.5 local threshold. That's the threshold, not this branch. Left alone; see Needs Bazil.

## Stage 3: rerank with qwen3 (done, off by default)

`rag/rerank.py`, turned on with `RAG_RERANK=1` or `eval.py --rerank`.

- The top 12 fused candidates (each cut to 600 characters, labelled "Note > Section") go to qwen3:8b in one call, with Ollama structured output (`format` = a JSON schema for `{"ranking": [int]}`), `think: false`, temperature 0.
- Passages the model ranks go first; the rest keep their fused order after them, so nothing is lost. Out-of-range, repeated or non-integer numbers are ignored.
- Any failure (timeout 60 s, bad JSON, no ranking) logs a warning and keeps the fused order, so rerank can't stop an answer.
- Only the Ollama provider has `chat_json`. With the cloud provider, rerank is skipped silently. Sending candidates to Claude for reranking would be easy to add but costs a call per question; I didn't.

Measured (retrieval only, warm qwen3, 57 questions):

| | Hybrid defaults | + rerank | Vector + rerank |
|---|---|---|---|
| Recall | 100% | 100% | 98% |
| @3 | 97% | 98% | 96% |
| MRR | 0.97 | 0.98 | 0.96 |
| Hit@1 | 95% | 96% | 95% |
| Retrieval p50 / p90 | 15 / 18 ms | 608 / 801 ms | 578 / 796 ms |

One question changed: "Who gave me the tomato seedlings?" moved Tom Okafor from rank 2 to 1. No fallbacks fired. Vector + rerank catches up with hybrid on ranking, but it can't fix the filter questions, because the right note never makes it into the vector candidates.

**Decision: keep it, off by default.** One question better for about 40 times the retrieval time (and several seconds more when qwen3 is cold) isn't worth it as a default on this set. It may earn its place on a real vault with more near-duplicates; try `python eval.py --provider local --rerank` on your own questions.

## Self-review

A fresh reviewer read `git diff origin/web-and-local...HEAD` cold. It found nothing wrong in the Postgres SQL (it checked placeholders, the generated column's immutability, DISTINCT ON ordering and migrating a `chunks` table that already has rows). It found six real problems. All are fixed, each with a test:

1. Inline filters kept trailing punctuation: `in:Kitchen?` looked for a folder called "Kitchen?" and found nothing. Bare values now stop before `?.,;:!)`, and the leftover " ?" is tidied up.
2. A short intro merged into the next section gave that chunk the note title as its heading instead of the section's. Carried words now take the heading of the section they join.
3. The folder filter matched any folder in the absolute path, so with the vault under `~/Documents`, `in:Documents` matched everything. Folders now match from the vault root down (`retrieve(..., vault_root=)`, passed from settings). Without a root it falls back to matching a path segment.
4. `#tag` filters only saw frontmatter tags. Inline `#tags` in the body (outside code) are now stored too.
5. Postgres with HNSW applies the filter after the index scan, which returns only 40 candidates by default, so filtering on a small tag or folder could find almost nothing. A filtered vector search now runs `SET LOCAL hnsw.ef_search = 400` first. **Not executed** (no database here).
6. The eval summary crashed when no question had expected sources. Fixed.

The Postgres tag-prefix check used `LIKE`, so `_` in a tag acted as a wildcard. It now uses `LEFT(...) = want || '/'`.

Final run after the fixes (new defaults, with answers): recall 100%, @3 97%, MRR 0.97, hit@1 95%, answer keywords 98%, median retrieval 16 ms, median answer 723 ms.

Seen in the screenshot and left as is (these predate this branch):

- `_clean_line` strips `_`, so `water_temp_c` is stored as `watertempc`. A keyword search for the exact column name misses it; the vector side still found it.
- Snippets start with "Tags: ..." because tags are part of the chunk text.
- Source cards are in fused order, so the "% match" numbers aren't always descending.

## Checks run

- `python3 -m unittest discover -s tests -t .`: 91 tests, OK (3 live tests skipped). That's 41 more than at the base.
- `python3 -m tests.live`: 5 tests, OK, against Ollama on pc (including a real qwen3 rerank).
- `cd web && npm ci && npm run build`: clean.
- Screenshots of the web app on the fixture vault, asking "Which notes link to the Lighthouse App?" (cards 7 and 8 show "Links here"):
  - `/data/tmp/overnight-2026-10-29/scratch/f-rag-hybrid/screenshots/desktop-2-answer.png`
  - `/data/tmp/overnight-2026-10-29/scratch/f-rag-hybrid/screenshots/phone-2b-sources.png`
  - plus the empty and preview shots in the same folder.
- Eval logs and JSON for every run quoted above: `/data/tmp/overnight-2026-10-29/scratch/f-rag-hybrid/eval/`.

## How to try it

```bash
# fixture eval, new defaults (hybrid, heading chunks, backlinks)
python3 eval.py --provider local --vault tests/fixtures/vault --memory --questions tests/fixtures/questions.json

# compare against the old behaviour, or try rerank
python3 eval.py ... --retrieval vector --chunking window --no-backlinks
python3 eval.py ... --rerank

# web app with filters: ask "#book which one did I read in a weekend?"
python3 server.py --provider local --vault tests/fixtures/vault --memory
```

## Cleanup

I stopped the API server (port 8797) I started. No containers were created (docker was held). The scratch venv at `scratch/f-rag-hybrid/venv` is empty; pip was held.

## Needs Bazil

1. **Run the Postgres path once.** Docker and pip were held again, so `setup()`'s new `tags` and `tsv` columns, the GIN index, `keyword_search`, `backlink_chunks` and the `ef_search` widening have only been read, never run. On the real `chunks` table, adding the generated `tsv` column rewrites the table once (a short lock). Try it on a scratch database first.
2. **One re-index when you upgrade.** The default chunking is now `heading`, so the first `ingest.py` re-embeds every note once. For cloud, that's Voyage calls for the whole vault. A: accept it · B: set `RAG_CHUNKING=window` to keep the old chunks, then switch when convenient.
3. **Run the eval on your real questions** with `--retrieval vector --chunking window --no-backlinks` and then with the defaults. The fixture vault is small and made up, so the gains here are a direction, not a promise. Add `expected_sources` to your `eval/questions.json` to get the source metrics.
4. **Rerank on your vault?** Off by default (+600 ms for one question gained here). Try `--rerank` on your questions. A: leave it off · B: turn it on if it helps there.
5. **Abstain is 0%.** Off-topic questions ("passport number") still score above the 0.5 local threshold and get an answer built from loosely related notes. That's a threshold or prompt question for another day.
6. **Cloud baseline wasn't run** (no keys here). Run the eval with `--provider cloud` once to record it.

## Decisions applied 2026-10-30

Overnight run on `pc`, local branch `ov/rag`, pushed to `hybrid-retrieval`. Decision numbers are from the 2026-10-29 morning digest.

### Decision 36A: accept that the first ingest re-embeds every note (no code change)

The README's ingest step now says so: the first ingest after upgrading re-embeds every note once (the stored hash records the chunking mode), which on the cloud provider means Voyage calls for the whole vault; later ingests skip unchanged notes; `RAG_CHUNKING=window` puts it off. Seen for real tonight: on a table filled by `web-and-local`, this branch's ingest rewrote all 26 notes (28 chunks became 41), and a second run wrote 0. Commit `b0aede3`.

### Decision 37A: rerank stays off by default (no change)

`RAG_RERANK` still defaults to `0`. Nothing changed.

### Postgres: first real run

A scratch `pgvector/pgvector:pg16` container (`rag-ov-rag-pg`, port 55471), psycopg2 from `uv run --with psycopg2-binary`, fixture vault only, local provider (Ollama on pc).

**Migration on tables that already have data.** Two cases, both with rows in them before this branch's `setup()` ran:

- `chunks_nomic_768` filled by the real `web-and-local` code (exported with `git archive`, ingested the fixture vault: 28 rows).
- A `chunks` table in `main`'s schema (1024 dims, no `heading` column, ivfflat index), 28 rows with the fixture text.

On both, `setup()` added `heading`, `tags` and the generated `tsv` column, filled `tsv` for every existing row (28/28, none empty), set `tags` to `{}`, built the GIN index, and running it a second time changed nothing. Keyword search on the migrated rows worked straight away (`gochugaru` finds Kimchi), with no re-embedding. The `ADD COLUMN ... GENERATED ... STORED` rewrite was instant at this size.

**What broke, and the fixes** (commit `40fcaa7` plus a guard in the final commit):

1. **Common words drowned out rare ones in Postgres keyword search.** `ts_rank` has no IDF, and the english stemmer turns "weekly" into "week", which is in every section heading of the six-week course note. For "What questions do I ask in the weekly review?" the keyword list was Weekly Review and then seven course sections; the memory store's BM25 gave Weekly Review and the Harbour Authority call. Each query word now gets its own tsquery, weighted by its BM25 IDF over the table (`ln(1 + (N - df + 0.5) / (df + 0.5))`, counts via the GIN index), and the per-word `ts_rank(..., 1)` scores are summed. Still BM25-like, not exact BM25.
2. **`main`'s ivfflat index barely works.** `main` created it in `setup()` while `chunks` was empty, so it trained on nothing. Rebuilt the same way with the real nomic vectors, an index scan asked for 20 neighbours returned 1 to 5. On a vault-sized table the planner uses that index, so cloud vector search has probably been missing most matches. `setup()` now adds an HNSW index if the table has none and drops the ivfflat one: an index, not rows. If pgvector is older than 0.5 (no HNSW), it leaves the indexes alone. On Bazil's real `chunks` table the next ingest builds the HNSW index once (it takes a moment and blocks writes while it runs).
3. Nothing else failed: placeholders, `DISTINCT ON` in `backlink_chunks`, `SET LOCAL hnsw.ef_search` for filtered searches, tag and folder filters, `ingest_log`, all ran fine.

**New tests.** `tests/test_postgres.py`, 10 tests, skipped unless `RAG_TEST_PG_DSN` is set. They migrate filled tables in `main`'s and `web-and-local`'s schemas (and check vector search through the index with seq scans off), ingest the fixture vault, and check that Postgres matches the memory store on stats, links, backlinks, linked chunks and keyword ranking, plus filters, tags and hybrid retrieval. The IDF test and the ivfflat test both failed before the fixes.

**Eval on Postgres** (fixture vault, local provider, new defaults, 57 questions, with answers):

| | Memory store | Postgres before fix | Postgres after fix |
|---|---|---|---|
| Source recall / @3 | 100% / 97% | 100% / 97% | 100% / 97% |
| MRR / hit@1 | 0.97 / 95% | 0.97 / 95% | 0.97 / 95% |
| Keyword retrieval | 98% | 98% | 98% |
| Answer keywords | 96% | 93% | 97% |
| Chunks per question | 7.16 | 6.84 | 6.98 |
| Same source set as memory | | 44/57 | 47/57 |
| Same source order as memory | | 33/57 | 43/57 |
| Median retrieval / answer | 18 / 745 ms | 37 / 786 ms | 16 / 724 ms |

The ranking metrics were already at the ceiling on this small vault, so the fix shows up as Postgres agreeing with the measured memory store more often, not as a higher MRR. Answer keywords move by a question or two between runs (qwen3 noise). The 37 ms is from the first run, with the Postgres cache cold.

Left as is: the english stemmer still folds "weekly" into "week", so the weekly review question still pulls course sections into its top five on Postgres (they're also the vector side's next-best hits). That's the stemmer working as designed; the memory store's plural-only stemmer doesn't do it.

### Checks run

- `python3 -m unittest discover -s tests -t .`: 101 tests, OK (13 skipped: 3 live, 10 Postgres).
- `tests.test_postgres` against the scratch container: 10 tests, OK.
- `python3 -m tests.live`: 5 tests, OK, against Ollama on pc.
- Full eval on Postgres and on the memory store, as in the table. Logs and JSON: `/data/tmp/overnight-2026-10-30/scratch/rag/eval-*.{log,json}`.
- Sentence case (decision 3A): the web app has no uppercase or letter-spaced labels. No UI changed tonight.
- Self-review of tonight's diff: added the pgvector < 0.5 guard above. Nothing else found.

How to rerun the Postgres check:

```bash
docker run -d --name rag-pg-test -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag_test \
  -p 127.0.0.1:55432:5432 pgvector/pgvector:pg16
RAG_TEST_PG_DSN="host=127.0.0.1 port=55432 dbname=rag_test user=rag password=rag" \
  uv run --no-project --with psycopg2-binary python -m unittest tests.test_postgres
docker rm -f rag-pg-test
```

### Cleanup

Removed the container `rag-ov-rag-pg` (no volume was named, so its anonymous volume went with `docker rm -v`). No servers left running.

### Needs Bazil (updated 2026-10-30)

1. **Your cloud `chunks` table gets a new vector index on the next ingest.** The old ivfflat index is swapped for HNSW (an index only, rows stay). Expect the first ingest to pause while it builds. Cloud answers may find noticeably more matching notes afterwards; that's the fix, not a regression.
2. **Run the eval on your real questions** (still open from 2026-10-29), now that the Postgres path has run: `python eval.py` with the defaults, then `--retrieval vector --chunking window --no-backlinks`.
3. **Abstain is 0%** and **the cloud baseline wasn't run** (no keys here): both unchanged from 2026-10-29.
