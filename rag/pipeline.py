"""Ingest, retrieve and answer, independent of which provider or store is used."""

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .notes import (chunk_note, clean_markdown, embed_text, extract_links, file_hash, note_name,
                    note_tags, parse_filters, parse_frontmatter, read_note)
from .providers import ProviderUnavailable

MIN_NOTE_CHARS = 50

# Hybrid retrieval: how many candidates each ranker contributes before fusion,
# and the usual reciprocal rank fusion constant.
CANDIDATES_PER_RANKER = 20
RRF_K = 60
# A keyword-only hit (no vector match above the threshold) is kept when its
# cosine similarity is at most this far below the threshold. This keeps exact
# matches on rare words (codes, names) while off-topic questions still come
# back empty.
KEYWORD_SIMILARITY_MARGIN = 0.05

SYSTEM_PROMPT = """You are a personal knowledge assistant with access to the user's Obsidian notes.
Answer questions based on the provided context from their notes.
Be conversational and specific — reference actual content from the notes.
If the context doesn't contain enough information to answer, say so honestly.
Never make up information that isn't in the notes.
When you use a numbered passage from the context, cite it inline like [1]."""


def in_dot_folder(path, root) -> bool:
    """True for .obsidian, .trash and other hidden folders inside the vault."""
    try:
        parts = Path(path).relative_to(root).parts
    except ValueError:
        return False
    return any(part.startswith(".") for part in parts)


def markdown_files(vault_path) -> list[Path]:
    root = Path(vault_path)
    return sorted(p for p in root.rglob("*.md") if not in_dot_folder(p, root))


def index_hash(file_path, chunking="heading") -> str:
    """The stored hash: the file's hash plus the chunking mode, so switching
    modes re-indexes each note once instead of mixing chunk styles."""
    fhash = file_hash(file_path)
    return fhash if chunking == "window" else f"{fhash}:{chunking}"


def ingest_file(store, provider, file_path, chunking="heading") -> int:
    """Index one note. Returns the number of chunks written (0 if unchanged)."""
    path = str(file_path)
    raw = read_note(file_path)
    links = extract_links(raw)
    tags = note_tags(parse_frontmatter(raw)[0])

    if len(clean_markdown(embed_text(raw))) < MIN_NOTE_CHARS:
        if store.file_hash(path) is not None:
            store.delete_file(path)  # the note was emptied since last time
        return 0

    fhash = index_hash(file_path, chunking)
    if store.file_hash(path) == fhash:
        store.update_links(path, links, tags)
        return 0

    chunks = chunk_note(raw, mode=chunking, title=note_name(path))
    embeddings = provider.embed([c.index_text for c in chunks], kind="document")
    store.replace_file(path, fhash, chunks, embeddings, links, tags)
    return len(chunks)


def ingest_vault(store, provider, vault_path, log=print, chunking="heading") -> dict:
    started = datetime.now(timezone.utc)
    root = Path(vault_path).expanduser()
    files = markdown_files(root)
    log(f"📂 Found {len(files)} markdown files in vault")

    changed = 0
    total = 0
    failed = []
    for i, f in enumerate(files):
        try:
            count = ingest_file(store, provider, f, chunking)
        except ProviderUnavailable:
            raise  # the backend is down; every other file would fail too
        except Exception as e:  # e.g. a note deleted mid-run; skip it, keep going
            failed.append(f.name)
            log(f"  ⚠️ Skipped {f.name}: {e}")
            continue
        if count:
            changed += 1
            log(f"  [{i+1}/{len(files)}] {f.name} → {count} chunks")
        total += count

    # Drop index rows for notes that were deleted or renamed, but only inside
    # this vault so another vault sharing the table is left alone.
    present = {str(f) for f in files}
    prefix = str(root) + os.sep
    removed = [p for p in store.file_paths() if p.startswith(prefix) and p not in present]
    for p in removed:
        store.delete_file(p)

    store.log_ingest(started, len(files), changed, total, len(removed))
    summary = {"files": len(files), "changed": changed, "chunks": total, "removed": len(removed),
               "failed": failed}
    log(f"\n✅ Done. {total} chunks ingested, {len(removed)} removed notes cleared.")
    return summary


def fuse(rankings, k=RRF_K) -> list[dict]:
    """Reciprocal rank fusion: each row scores sum(1 / (k + rank)) over the
    rankings it appears in. Rows are matched on (file, chunk_index)."""
    scores: dict[tuple, float] = {}
    rows: dict[tuple, dict] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            key = (row["file"], row["chunk_index"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            rows.setdefault(key, row)
    order = sorted(scores, key=lambda key: -scores[key])  # stable: earlier rankings win ties
    return [{**rows[key], "score": scores[key]} for key in order]


def merge_filters(*filters) -> dict:
    """Combine filter dicts: tags add up, the last folder wins."""
    out: dict = {}
    for f in filters:
        if not f:
            continue
        if f.get("tags"):
            out["tags"] = list(dict.fromkeys([*out.get("tags", []), *f["tags"]]))
        if f.get("folder"):
            out["folder"] = f["folder"]
    return out


def retrieve(store, provider, query, top_k=5, min_similarity=0.35, *, mode="vector",
             filters=None, backlinks=False, rerank=False) -> list[dict]:
    """Find the chunks to answer `query` from.

    mode="vector": nearest chunks above `min_similarity`.
    mode="hybrid": vector and keyword rankings merged with reciprocal rank fusion.
    `filters` ({"tags": [...], "folder": "..."}) and inline filters in the
    query (#tag, tag:x, in:Folder) limit the primary hits. Then the notes the
    hits link to are added, and with `backlinks`, notes that link to the hits.
    `rerank` reorders the candidates with the chat model before the cut.
    """
    if mode not in ("hybrid", "vector"):
        raise ValueError(f"mode must be 'hybrid' or 'vector', got {mode!r}")
    text, inline = parse_filters(query)
    filters = merge_filters(inline, filters)
    query_embedding = provider.embed([text], kind="query")[0]
    depth = max(top_k, CANDIDATES_PER_RANKER) if (mode == "hybrid" or rerank) else top_k
    candidates = store.search(query_embedding, depth, min_similarity, filters)
    if mode == "hybrid":
        floor = min_similarity - KEYWORD_SIMILARITY_MARGIN
        keyword_rows = [r for r in store.keyword_search(text, depth, filters, embedding=query_embedding)
                        if r["similarity"] > floor]
        candidates = fuse([candidates, keyword_rows])
    if rerank and len(candidates) > 1:
        from .rerank import rerank_rows

        candidates = rerank_rows(provider, text, candidates)
    primary = candidates[:top_k]
    if not primary:
        return []

    linked_names = []
    for row in primary:
        for link in row["links"]:
            if link.lower() not in {n.lower() for n in linked_names}:
                linked_names.append(link)
    primary_files = {row["file"] for row in primary}
    linked = store.chunks_for_names(linked_names, exclude_files=primary_files)
    linking = []
    if backlinks:
        names = list(dict.fromkeys(note_name(row["file"]) for row in primary))
        exclude = primary_files | {row["file"] for row in linked}
        linking = store.backlink_chunks(names, exclude_files=exclude)

    seen = set()
    results = []
    for via, rows in ((None, primary), ("link", linked), ("backlink", linking)):
        for row in rows:
            key = (row["file"], row["content"][:50])
            if key in seen:
                continue
            seen.add(key)
            results.append({**row, "via_link": via is not None, "via": via})
    return results


def build_context(chunks) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        filename = os.path.basename(chunk["file"])
        where = f"'{filename}'" + (f", section '{chunk['heading']}'" if chunk.get("heading") else "")
        if chunk.get("via") == "backlink":
            label = "a note that links to one of the passages above"
        elif chunk.get("via_link"):
            label = "linked note"
        else:
            label = f"similarity: {chunk['similarity']:.2f}"
        parts.append(f"[{i+1}] From {where} ({label}):\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def user_message(query, chunks) -> str:
    return f"""Here is relevant context from my notes:

{build_context(chunks)}

---

My question: {query}"""


def stream_answer(provider, query, chunks, history=None) -> Iterator[str]:
    messages = list(history or [])
    messages.append({"role": "user", "content": user_message(query, chunks)})
    yield from provider.stream_chat(SYSTEM_PROMPT, messages)


def answer(provider, query, chunks, history=None) -> str:
    text = "".join(stream_answer(provider, query, chunks, history))
    if history is not None:
        history.append({"role": "user", "content": user_message(query, chunks)})
        history.append({"role": "assistant", "content": text})
    return text


class Timer:
    def __init__(self):
        self.start = time.perf_counter()

    def ms(self) -> int:
        return round((time.perf_counter() - self.start) * 1000)
