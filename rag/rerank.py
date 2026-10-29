"""Optional rerank: the local chat model reorders the retrieval candidates.

Off by default (RAG_RERANK=1 turns it on). Only the Ollama provider supports
it; with any other provider the candidates keep their order. If the model
call fails or returns nonsense, the original order is kept, so a rerank
problem never stops an answer.
"""

import logging

from .notes import note_name

log = logging.getLogger(__name__)

RERANK_CANDIDATES = 12  # how many of the fused candidates the model sees
PASSAGE_CHARS = 600  # each passage is cut to this, to keep the prompt small
RERANK_TIMEOUT = 60.0

SYSTEM = """You rank passages from a person's notes by how useful they are for answering their question.
Judge only relevance to the question. A passage that contains the answer ranks first."""

SCHEMA = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
    "required": ["ranking"],
}


def _prompt(query, rows):
    parts = []
    for i, row in enumerate(rows, start=1):
        where = note_name(row["file"])
        if row.get("heading") and row["heading"] != where:
            where += f" > {row['heading']}"
        text = row["content"]
        if len(text) > PASSAGE_CHARS:
            text = text[:PASSAGE_CHARS].rsplit(" ", 1)[0] + " ..."
        parts.append(f"[{i}] {where}\n{text}")
    return (f"Question: {query}\n\nPassages:\n\n" + "\n\n".join(parts) +
            f"\n\nReturn JSON with \"ranking\": the passage numbers (1 to {len(rows)}) "
            "from most to least relevant. Leave out passages that don't help at all.")


def rerank_rows(provider, query, rows):
    """Reorder `rows` with the model. Rows the model leaves out go after the
    ones it ranked, in their original order, so nothing is lost."""
    chat_json = getattr(provider, "chat_json", None)
    if chat_json is None or len(rows) < 2:
        return rows
    head, tail = rows[:RERANK_CANDIDATES], rows[RERANK_CANDIDATES:]
    try:
        reply = chat_json(SYSTEM, _prompt(query, head), SCHEMA, timeout=RERANK_TIMEOUT)
        ranking = reply.get("ranking") if isinstance(reply, dict) else None
        if not isinstance(ranking, list):
            raise ValueError(f"no ranking in {reply!r:.200}")
    except Exception as e:  # noqa: BLE001 - any failure falls back to the fused order
        log.warning("Rerank skipped: %s", e)
        return rows
    order = []
    for n in ranking:
        if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(head) and n - 1 not in order:
            order.append(n - 1)
    ranked = [head[i] for i in order]
    rest = [row for i, row in enumerate(head) if i not in order]
    return [{**row, "reranked": True} for row in ranked] + rest + tail
