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

## Status

All three stages done. Self-review next.
