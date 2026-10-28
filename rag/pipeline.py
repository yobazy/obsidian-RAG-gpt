"""Ingest, retrieve and answer, independent of which provider or store is used."""

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .notes import chunk_note, clean_markdown, extract_links, file_hash, read_note

MIN_NOTE_CHARS = 50

SYSTEM_PROMPT = """You are a personal knowledge assistant with access to the user's Obsidian notes.
Answer questions based on the provided context from their notes.
Be conversational and specific — reference actual content from the notes.
If the context doesn't contain enough information to answer, say so honestly.
Never make up information that isn't in the notes."""


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


def ingest_file(store, provider, file_path) -> int:
    """Index one note. Returns the number of chunks written (0 if unchanged)."""
    path = str(file_path)
    raw = read_note(file_path)
    links = extract_links(raw)

    if len(clean_markdown(raw)) < MIN_NOTE_CHARS:
        if store.file_hash(path) is not None:
            store.delete_file(path)  # the note was emptied since last time
        return 0

    fhash = file_hash(file_path)
    if store.file_hash(path) == fhash:
        store.update_links(path, links)
        return 0

    chunks = chunk_note(raw)
    embeddings = provider.embed([c.content for c in chunks], kind="document")
    store.replace_file(path, fhash, chunks, embeddings, links)
    return len(chunks)


def ingest_vault(store, provider, vault_path, log=print) -> dict:
    started = datetime.now(timezone.utc)
    root = Path(vault_path).expanduser()
    files = markdown_files(root)
    log(f"📂 Found {len(files)} markdown files in vault")

    changed = 0
    total = 0
    for i, f in enumerate(files):
        count = ingest_file(store, provider, f)
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
    summary = {"files": len(files), "changed": changed, "chunks": total, "removed": len(removed)}
    log(f"\n✅ Done. {total} chunks ingested, {len(removed)} removed notes cleared.")
    return summary


def retrieve(store, provider, query, top_k=5, min_similarity=0.35) -> list[dict]:
    """Embed the query, find similar chunks, expand via wikilinks."""
    query_embedding = provider.embed([query], kind="query")[0]
    primary = store.search(query_embedding, top_k, min_similarity)
    if not primary:
        return []

    linked_names = []
    for row in primary:
        for link in row["links"]:
            if link.lower() not in {n.lower() for n in linked_names}:
                linked_names.append(link)
    primary_files = {row["file"] for row in primary}
    linked = store.chunks_for_names(linked_names, exclude_files=primary_files)

    seen = set()
    results = []
    for via_link, rows in ((False, primary), (True, linked)):
        for row in rows:
            key = (row["file"], row["content"][:50])
            if key in seen:
                continue
            seen.add(key)
            results.append({**row, "via_link": via_link})
    return results


def build_context(chunks) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        filename = os.path.basename(chunk["file"])
        where = f"'{filename}'" + (f", section '{chunk['heading']}'" if chunk.get("heading") else "")
        label = "linked note" if chunk.get("via_link") else f"similarity: {chunk['similarity']:.2f}"
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
