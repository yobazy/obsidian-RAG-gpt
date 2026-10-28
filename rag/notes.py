"""Reading Obsidian notes: frontmatter, wikilinks, cleaning and chunking."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_WIKILINK = re.compile(r"(!?)\[\[([^\]]+)\]\]")
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(\n|\Z)", re.DOTALL)


@dataclass
class Chunk:
    index: int
    content: str
    heading: str | None


def file_hash(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def read_note(path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def note_name(path) -> str:
    """The name Obsidian links to: the file name without .md."""
    name = Path(path).name
    return name[:-3] if name.lower().endswith(".md") else name


def extract_links(text: str) -> list[str]:
    """Names of notes this text links to via [[wikilinks]].

    `[[Folder/Note#Heading|alias]]` becomes `Note`. Embeds of non-note files
    (`![[diagram.png]]`) are skipped. Order is kept, duplicates dropped.
    """
    seen = {}
    for bang, inner in _WIKILINK.findall(_CODE_BLOCK.sub("", text)):
        target = inner.split("|", 1)[0].split("#", 1)[0].strip()
        if not target:
            continue
        target = target.rsplit("/", 1)[-1]
        suffix = Path(target).suffix.lower()
        if suffix and suffix != ".md":
            if bang:
                continue  # an embedded image or PDF, not a note
        if suffix == ".md":
            target = target[:-3]
        seen.setdefault(target, None)
    return list(seen)


def _wikilink_text(match: re.Match) -> str:
    bang, inner = match.group(1), match.group(2)
    if bang:
        return ""
    target, _, alias = inner.partition("|")
    if alias.strip():
        return alias.strip()
    return target.split("#", 1)[0].rsplit("/", 1)[-1].strip()


def _clean_line(line: str) -> str:
    line = re.sub(r"#+ ", "", line)
    line = _WIKILINK.sub(_wikilink_text, line)
    line = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", line)
    line = re.sub(r"[*_`>-]", "", line)
    return line


def clean_markdown(text: str) -> str:
    """Strip markdown syntax for cleaner embeddings. Link text is kept."""
    text = _CODE_BLOCK.sub("", text)
    return "\n".join(_clean_line(line) for line in text.split("\n")).strip()


def _words_with_headings(text: str):
    """Yield (word, heading in effect) for the cleaned text."""
    heading = None
    for line in _CODE_BLOCK.sub("", text).split("\n"):
        match = _HEADING.match(line)
        if match:
            heading = _clean_line(match.group(2)).strip() or heading
        for word in _clean_line(line).split():
            yield word, heading


def chunk_note(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[Chunk]:
    """Overlapping word windows over the cleaned note.

    Each chunk carries the heading in effect where it starts, so answers can
    cite a section and not only a file.
    """
    pairs = list(_words_with_headings(text))
    chunks = []
    start = 0
    while start < len(pairs):
        end = min(start + chunk_size, len(pairs))
        window = pairs[start:end]
        heading = window[0][1]
        chunks.append(Chunk(len(chunks), " ".join(w for w, _ in window), heading))
        if end == len(pairs):
            break
        start += chunk_size - overlap
    return chunks


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a note into (frontmatter dict, body). Handles the simple YAML
    Obsidian writes: `key: value`, `key: [a, b]` and `- item` lists."""
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    data: dict = {}
    current = None
    for raw in match.group(1).split("\n"):
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = re.match(r"^\s*-\s+(.*)$", line)
        if item and current is not None:
            if not isinstance(data.get(current), list):
                data[current] = []
            data[current].append(_scalar(item.group(1)))
            continue
        pair = re.match(r"^([^:\s][^:]*):\s*(.*)$", line)
        if not pair:
            continue
        current = pair.group(1).strip()
        value = pair.group(2).strip()
        if value.startswith("[") and value.endswith("]"):
            data[current] = [_scalar(v) for v in value[1:-1].split(",") if v.strip()]
        elif value == "":
            data[current] = None
        else:
            data[current] = _scalar(value)
    return data, text[match.end():]


def _scalar(value: str):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value
