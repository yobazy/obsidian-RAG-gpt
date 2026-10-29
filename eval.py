import argparse
import json
import os
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rag.config import add_common_args, settings_from_args
from rag.notes import note_name
from rag.pipeline import Timer, answer, ingest_vault, retrieve
from rag.providers import ProviderUnavailable, make_provider
from rag.store import MemoryStore, make_store

console = Console()
DEFAULT_QUESTIONS = Path(__file__).parent / "eval" / "questions.json"


def normalize(text):
    """Lowercase and strip punctuation for fuzzy keyword matching."""
    return re.sub(r"[^\w\s]", "", text.lower())


def score_keywords(keywords, text):
    """Return (hits, misses, score) for expected keywords in text."""
    normalized_text = normalize(text)
    compact_text = normalized_text.replace(" ", "")

    hits = []
    misses = []
    for keyword in keywords:
        normalized_keyword = normalize(keyword)
        compact_keyword = normalized_keyword.replace(" ", "")
        if (
            normalized_keyword in normalized_text
            or compact_keyword in compact_text
        ):
            hits.append(keyword)
        else:
            misses.append(keyword)

    score = len(hits) / len(keywords) if keywords else 0.0
    return hits, misses, score


def score_sources(expected, retrieved):
    """Note-level retrieval metrics.

    `retrieved` is the list of note names in the order they reach the model
    (duplicates allowed). Returns recall (share of expected notes present),
    the same within the first three notes, reciprocal rank of the first expected note, and whether the top note is
    an expected one.
    """
    order = list(dict.fromkeys(name.lower() for name in retrieved))
    wanted = {name.lower() for name in expected}
    if not wanted:
        return {"recall": None, "recall3": None, "rr": None, "hit1": None}
    found = wanted & set(order)
    rank = next((i for i, name in enumerate(order, start=1) if name in wanted), None)
    return {
        "recall": len(found) / len(wanted),
        "recall3": len(wanted & set(order[:3])) / len(wanted),
        "rr": 1 / rank if rank else 0.0,
        "hit1": bool(order) and order[0] in wanted,
    }


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_eval(store, provider, settings, questions, retrieval_only=False):
    results = []

    for i, item in enumerate(questions, start=1):
        question = item["question"]
        keywords = item["expected_keywords"]

        console.print(f"\n[bold][{i}/{len(questions)}][/bold] {question}")

        timer = Timer()
        chunks = retrieve(store, provider, question, settings.top_k, settings.threshold)
        retrieval_ms = timer.ms()
        context_text = "\n".join(chunk["content"] for chunk in chunks)
        retrieval_hits, retrieval_misses, retrieval_score = score_keywords(
            keywords, context_text
        )
        expected_sources = item.get("expected_sources", [])
        source_scores = score_sources(expected_sources, [note_name(c["file"]) for c in chunks])
        abstained = None
        if item.get("type") == "abstain":
            abstained = not any(not c.get("via_link") for c in chunks)

        if not chunks:
            console.print("[red]  No chunks retrieved[/red]")
        else:
            sources = ", ".join(
                os.path.basename(chunk["file"]) for chunk in chunks[:3]
            )
            console.print(f"[dim]  Sources: {sources}[/dim]")

        if source_scores["recall"] is not None:
            console.print(
                f"  Sources:   [cyan]{source_scores['recall']:.0%}[/cyan] recall, "
                f"rank of first expected {round(1 / source_scores['rr']) if source_scores['rr'] else '-'}"
            )
        if abstained is not None:
            console.print(f"  Abstained: {'[green]yes' if abstained else '[red]no'}[/]")
        console.print(
            f"  Retrieval: [cyan]{retrieval_score:.0%}[/cyan] "
            f"({len(retrieval_hits)}/{len(keywords)} keywords)"
        )
        if retrieval_misses:
            console.print(f"  [dim]Missing in context: {', '.join(retrieval_misses)}[/dim]")

        answer_score = None
        answer_hits = []
        answer_misses = []
        answer_ms = None
        error = None
        if not retrieval_only:
            timer = Timer()
            try:
                response = answer(provider, question, chunks)
            except ProviderUnavailable as e:
                # Score it as a miss and keep going; one bad call shouldn't sink the run.
                console.print(f"  [red]Answer failed: {e}[/red]")
                response = ""
                error = str(e)
            answer_ms = timer.ms()
            answer_hits, answer_misses, answer_score = score_keywords(
                keywords, response
            )
            console.print(
                f"  Answer:    [cyan]{answer_score:.0%}[/cyan] "
                f"({len(answer_hits)}/{len(keywords)} keywords)"
            )
            console.print(f"  [dim]Latency: retrieval {retrieval_ms} ms, answer {answer_ms} ms[/dim]")
            if answer_misses:
                console.print(f"  [dim]Missing in answer: {', '.join(answer_misses)}[/dim]")

        results.append({
            "question": question,
            "type": item.get("type", "fact"),
            "expected_sources": expected_sources,
            **{f"source_{k}": v for k, v in source_scores.items()},
            "abstained": abstained,
            "retrieval_score": retrieval_score,
            "answer_score": answer_score,
            "retrieval_misses": retrieval_misses,
            "answer_misses": answer_misses,
            "chunk_count": len(chunks),
            "sources": [os.path.basename(chunk["file"]) for chunk in chunks],
            "retrieval_ms": retrieval_ms,
            "answer_ms": answer_ms,
            "error": error,
        })

    return results


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(results, retrieval_only=False):
    """Averages over the scored questions. Abstain cases (no expected notes)
    only count towards the abstain rate."""
    scored = [r for r in results if r["type"] != "abstain"]
    abstain = [r for r in results if r["type"] == "abstain"]
    summary = {
        "questions": len(results),
        "source_recall": _mean(r["source_recall"] for r in scored),
        "source_recall_at_3": _mean(r["source_recall3"] for r in scored),
        "source_mrr": _mean(r["source_rr"] for r in scored),
        "hit_at_1": _mean(r["source_hit1"] for r in scored),
        "all_sources_found": _mean(r["source_recall"] == 1.0 for r in scored if r["source_recall"] is not None),
        "keyword_retrieval": _mean(r["retrieval_score"] for r in scored),
        "abstain_rate": _mean(r["abstained"] for r in abstain),
        "avg_chunks": _mean(r["chunk_count"] for r in results),
        "median_retrieval_ms": median(r["retrieval_ms"] for r in results),
        "by_type": {},
    }
    if not retrieval_only:
        summary["answer"] = _mean(r["answer_score"] for r in scored)
        summary["median_answer_ms"] = median(r["answer_ms"] for r in results)
    for kind in dict.fromkeys(r["type"] for r in scored):
        group = [r for r in scored if r["type"] == kind]
        summary["by_type"][kind] = {
            "n": len(group),
            "source_recall": _mean(r["source_recall"] for r in group),
            "source_recall_at_3": _mean(r["source_recall3"] for r in group),
            "source_mrr": _mean(r["source_rr"] for r in group),
        }
    return summary


def _pct(value):
    return "-" if value is None else f"{value:.0%}"


def print_summary(results, retrieval_only=False):
    table = Table(title="Eval summary")
    table.add_column("#", style="dim")
    table.add_column("Type")
    table.add_column("Question", max_width=50)
    table.add_column("Source recall", justify="right")
    table.add_column("Keywords", justify="right")
    if not retrieval_only:
        table.add_column("Answer", justify="right")

    for i, result in enumerate(results, start=1):
        if result["type"] == "abstain":
            recall = "abstained" if result["abstained"] else "answered"
        else:
            recall = _pct(result["source_recall"])
        row = [str(i), result["type"], result["question"][:50], recall,
               "-" if result["type"] == "abstain" else _pct(result["retrieval_score"])]
        if not retrieval_only:
            row.append("-" if result["type"] == "abstain" else _pct(result["answer_score"]))
        table.add_row(*row)

    summary = summarize(results, retrieval_only)
    console.print()
    console.print(table)
    console.print(
        f"\n[bold]Source recall:[/bold] {_pct(summary['source_recall'])}  "
        f"[bold]@3:[/bold] {_pct(summary['source_recall_at_3'])}  "
        f"[bold]MRR:[/bold] {summary['source_mrr']:.2f}  "
        f"[bold]Hit@1:[/bold] {_pct(summary['hit_at_1'])}  "
        f"[bold]All sources found:[/bold] {_pct(summary['all_sources_found'])}"
    )
    console.print(f"[bold]Keyword retrieval:[/bold] {_pct(summary['keyword_retrieval'])}  "
                  f"[bold]Abstain:[/bold] {_pct(summary['abstain_rate'])}  "
                  f"[bold]Chunks per question:[/bold] {summary['avg_chunks']:.1f}")
    for kind, group in summary["by_type"].items():
        console.print(f"  {kind:<11} n={group['n']:<3} recall {_pct(group['source_recall'])}  "
                      f"@3 {_pct(group['source_recall_at_3'])}  "
                      f"MRR {group['source_mrr']:.2f}")
    if "answer" in summary:
        console.print(f"[bold]Answer keywords:[/bold] {_pct(summary['answer'])}")
    console.print(f"[bold]Median retrieval latency:[/bold] {summary['median_retrieval_ms']} ms")
    if "median_answer_ms" in summary:
        console.print(f"[bold]Median answer latency:[/bold]    {summary['median_answer_ms']} ms")
    return summary


def median(values):
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else round((values[mid - 1] + values[mid]) / 2)


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Run RAG eval against hand-written Q&A pairs"))
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Score retrieval only (skip LLM calls)",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Ingest the vault into an in-memory index first instead of using Postgres "
             "(handy for the small fixture vault)",
    )
    parser.add_argument("--json-out", type=Path, help="Also write per-question results as JSON")
    args = parser.parse_args()
    settings = settings_from_args(args)

    questions = load_questions(args.questions)
    console.print(
        f"\n[bold green]Eval harness[/bold green] — {len(questions)} questions, "
        f"{settings.provider} provider"
    )
    if args.retrieval_only:
        console.print("[dim]Retrieval-only mode (no LLM calls)[/dim]")

    provider = make_provider(settings)
    try:
        if args.memory:
            if not settings.vault_path:
                sys.exit("--memory needs VAULT_PATH or --vault")
            store = MemoryStore(settings)
            timer = Timer()
            ingest_vault(store, provider, settings.vault_root, log=lambda *_: None)
            console.print(f"[dim]Indexed {store.stats()['chunks']} chunks in {timer.ms()} ms[/dim]")
        else:
            store = make_store(settings)
            store.setup()
        results = run_eval(store, provider, settings, questions, retrieval_only=args.retrieval_only)
    except ProviderUnavailable as e:
        sys.exit(f"❌ {e}")
    summary = print_summary(results, retrieval_only=args.retrieval_only)
    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"provider": settings.provider, "summary": summary, "results": results}, indent=2))
    store.close()


if __name__ == "__main__":
    main()
