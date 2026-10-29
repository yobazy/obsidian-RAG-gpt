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

## Status

Stage 2 in progress.
