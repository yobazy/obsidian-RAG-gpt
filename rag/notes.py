"""Reading Obsidian notes: frontmatter, wikilinks, cleaning and chunking."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
# Heading-aware chunking: sections shorter than this many words are merged
# into the next one, so a heading with one line under it isn't a chunk alone.
MIN_SECTION_WORDS = 25
CHUNKING_MODES = ("heading", "window")

_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_WIKILINK = re.compile(r"(!?)\[\[([^\]]+)\]\]")
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(\n|\Z)", re.DOTALL)


@dataclass
class Chunk:
    index: int
    content: str
    heading: str | None
    # What gets embedded and keyword-indexed: the content with the note title
    # and section in front, so a section chunk still says what note it is from.
    # Empty means the same as content.
    context: str = ""

    @property
    def index_text(self) -> str:
        return self.context or self.content


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


def embed_text(text: str) -> str:
    """The part of a note worth embedding: the body, with frontmatter tags
    kept as one plain line and other properties dropped."""
    meta, body = parse_frontmatter(text)
    tags = meta.get("tags")
    if isinstance(tags, str):
        tags = [tags]
    if tags:
        body = "Tags: " + ", ".join(str(t).lstrip("#") for t in tags) + "\n" + body
    return body


def chunk_note(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
               mode="heading", title=None) -> list[Chunk]:
    """Split a note into chunks.

    `window`: overlapping word windows over the whole note. Each chunk carries
    the heading in effect where it starts (or, for the opening chunk, the
    first heading inside it).

    `heading`: one chunk per section (split at every heading), with long
    sections cut into the same overlapping windows and short ones merged
    into the next section. Each chunk's index text starts with the note title
    and section, e.g. "Weekend Loaf > Lessons".
    """
    pairs = list(_words_with_headings(embed_text(text)))
    if mode == "window":
        return _windows(pairs, chunk_size, overlap)
    if mode != "heading":
        raise ValueError(f"chunking must be one of {CHUNKING_MODES}, got {mode!r}")

    if len(pairs) <= chunk_size:
        # A note that fits in one chunk stays whole: splitting a short note
        # only crowds the top results with several pieces of it (measured,
        # see docs/PROGRESS-hybrid.md). No title prefix either: it measured
        # worse, and the note's own "# Title" line is usually in the text.
        return _windows(pairs, chunk_size, overlap)

    sections: list[list[tuple[str, str | None]]] = []
    for pair in pairs:
        if not sections or pair[1] != sections[-1][-1][1]:
            sections.append([])
        sections[-1].append(pair)
    merged: list[list[tuple[str, str | None]]] = []
    carry: list = []
    for section in sections:
        section = carry + section
        if len(section) < MIN_SECTION_WORDS:
            carry = section
            continue
        carry = []
        merged.append(section)
    if carry:
        if merged:
            merged[-1].extend(carry)
        else:
            merged.append(carry)

    chunks = []
    for section in merged:
        for piece in _windows(section, chunk_size, overlap):
            label = title or ""
            if piece.heading and piece.heading != title:
                label = f"{title} > {piece.heading}" if title else piece.heading
            chunks.append(Chunk(len(chunks), piece.content, piece.heading,
                                f"{label}\n{piece.content}" if label else ""))
    return chunks


def _windows(pairs, chunk_size, overlap) -> list[Chunk]:
    chunks = []
    start = 0
    while start < len(pairs):
        end = min(start + chunk_size, len(pairs))
        window = pairs[start:end]
        heading = window[0][1] or next((h for _, h in window if h), None)
        chunks.append(Chunk(len(chunks), " ".join(w for w, _ in window), heading))
        if end == len(pairs):
            break
        start += chunk_size - overlap
    return chunks


def note_tags(meta: dict) -> list[str]:
    """Frontmatter tags, lowercased, without a leading #."""
    tags = meta.get("tags") or meta.get("tag") or []
    if isinstance(tags, str):
        tags = [t for t in re.split(r"[,\s]+", tags) if t]
    return [str(t).strip().lstrip("#").lower() for t in tags if str(t).strip().lstrip("#")]


# Inline filters in a question: #tag, tag:name, in:Folder, folder:"Two Words".
_FILTER = re.compile(
    r"""(?:(?<=\s)|^)(?:\#(?P<hash>[A-Za-z][\w/-]*)"""
    r"""|(?P<key>tag|in|folder):(?:"(?P<quoted>[^"]+)"|(?P<bare>[^\s"]+)))"""
)


def parse_filters(query: str) -> tuple[str, dict]:
    """Pull inline filters out of a question.

    Returns (question without the filters, {"tags": [...], "folder": "..."}).
    Tags match any-of and include nested tags (#book matches #book/fiction).
    """
    tags, folder = [], None

    def take(match):
        nonlocal folder
        if match.group("hash"):
            tags.append(match.group("hash").lower())
        elif match.group("key") == "tag":
            tags.append((match.group("quoted") or match.group("bare")).lstrip("#").lower())
        else:
            folder = (match.group("quoted") or match.group("bare")).strip("/")
        return ""

    cleaned = re.sub(r"\s{2,}", " ", _FILTER.sub(take, query)).strip()
    filters = {}
    if tags:
        filters["tags"] = tags
    if folder:
        filters["folder"] = folder
    return (cleaned or query), filters


def matches_filters(path: str, tags, filters: dict | None) -> bool:
    """Whether a note passes {"tags": [...], "folder": "..."} (both optional)."""
    if not filters:
        return True
    wanted = [t.lstrip("#").lower() for t in filters.get("tags") or []]
    if wanted:
        have = {t.lower() for t in tags or []}
        if not any(t in have or any(h.startswith(t + "/") for h in have) for t in wanted):
            return False
    folder = (filters.get("folder") or "").strip("/")
    if folder and f"/{folder.lower()}/" not in "/" + path.replace("\\", "/").lower():
        return False
    return True


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
