import argparse
import json
import os
import re
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from chat import DB_CONFIG, answer, retrieve

load_dotenv()

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


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_eval(conn, questions, retrieval_only=False):
    results = []

    for i, item in enumerate(questions, start=1):
        question = item["question"]
        keywords = item["expected_keywords"]

        console.print(f"\n[bold][{i}/{len(questions)}][/bold] {question}")

        chunks = retrieve(conn, question)
        context_text = "\n".join(chunk["content"] for chunk in chunks)
        retrieval_hits, retrieval_misses, retrieval_score = score_keywords(
            keywords, context_text
        )

        if not chunks:
            console.print("[red]  No chunks retrieved[/red]")
        else:
            sources = ", ".join(
                os.path.basename(chunk["file"]) for chunk in chunks[:3]
            )
            console.print(f"[dim]  Sources: {sources}[/dim]")

        console.print(
            f"  Retrieval: [cyan]{retrieval_score:.0%}[/cyan] "
            f"({len(retrieval_hits)}/{len(keywords)} keywords)"
        )
        if retrieval_misses:
            console.print(f"  [dim]Missing in context: {', '.join(retrieval_misses)}[/dim]")

        answer_score = None
        answer_hits = []
        answer_misses = []
        if not retrieval_only:
            response = answer(question, chunks)
            answer_hits, answer_misses, answer_score = score_keywords(
                keywords, response
            )
            console.print(
                f"  Answer:    [cyan]{answer_score:.0%}[/cyan] "
                f"({len(answer_hits)}/{len(keywords)} keywords)"
            )
            if answer_misses:
                console.print(f"  [dim]Missing in answer: {', '.join(answer_misses)}[/dim]")

        results.append({
            "question": question,
            "retrieval_score": retrieval_score,
            "answer_score": answer_score,
            "retrieval_misses": retrieval_misses,
            "answer_misses": answer_misses,
            "chunk_count": len(chunks),
        })

    return results


def print_summary(results, retrieval_only=False):
    avg_retrieval = sum(r["retrieval_score"] for r in results) / len(results)
    avg_answer = None
    if not retrieval_only:
        avg_answer = sum(r["answer_score"] for r in results) / len(results)

    table = Table(title="Eval Summary")
    table.add_column("#", style="dim")
    table.add_column("Question", max_width=50)
    table.add_column("Retrieval", justify="right")
    if not retrieval_only:
        table.add_column("Answer", justify="right")

    for i, result in enumerate(results, start=1):
        row = [
            str(i),
            result["question"][:50],
            f"{result['retrieval_score']:.0%}",
        ]
        if not retrieval_only:
            row.append(f"{result['answer_score']:.0%}")
        table.add_row(*row)

    console.print()
    console.print(table)
    console.print(
        f"\n[bold]Average retrieval:[/bold] {avg_retrieval:.0%} "
        f"({sum(1 for r in results if r['retrieval_score'] == 1.0)}/{len(results)} perfect)"
    )
    if avg_answer is not None:
        console.print(
            f"[bold]Average answer:[/bold]    {avg_answer:.0%} "
            f"({sum(1 for r in results if r['answer_score'] == 1.0)}/{len(results)} perfect)"
        )


def main():
    parser = argparse.ArgumentParser(description="Run RAG eval against hand-written Q&A pairs")
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Score retrieval only (skip Claude API calls)",
    )
    args = parser.parse_args()

    questions = load_questions(args.questions)
    console.print(
        f"\n[bold green]Eval harness[/bold green] — {len(questions)} questions"
    )
    if args.retrieval_only:
        console.print("[dim]Retrieval-only mode (no LLM calls)[/dim]")

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        results = run_eval(conn, questions, retrieval_only=args.retrieval_only)
        print_summary(results, retrieval_only=args.retrieval_only)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
