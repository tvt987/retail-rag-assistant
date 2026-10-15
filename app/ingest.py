"""Document parsing and text preprocessing that run before indexing.

Markdown, plain text and HTML are parsed with the standard library, so the
default offline path stays dependency-free. PDF parsing is optional and needs
`pypdf` (see requirements-extras.txt).
"""
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
import hashlib
import re
import unicodedata

SUPPORTED = (".md", ".markdown", ".txt", ".html", ".htm", ".pdf")


class IngestError(RuntimeError):
    """A file could not be parsed into text."""


# ---------------------------------------------------------------- parsing

_MD_RULES = [
    (re.compile(r"^---\n.*?\n---\n", re.S), ""),            # YAML front matter
    (re.compile(r"```.*?```", re.S), ""),                     # fenced code blocks
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),           # images -> alt text
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),            # links -> link text
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),             # heading markers
    (re.compile(r"^\s*>\s?", re.M), ""),                      # block quotes
    (re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.M), ""),      # list markers
    (re.compile(r"(\*\*|__|\*|_|`)(\S(?:.*?\S)?)\1"), r"\2"),  # emphasis, inline code
]


def markdown_to_text(markdown: str) -> str:
    text = markdown
    for pattern, replacement in _MD_RULES:
        text = pattern.sub(replacement, text)
    return text


class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "section", "article", "li", "ul", "ol", "table", "tr",
             "h1", "h2", "h3", "h4", "h5", "h6", "br", "header", "footer", "main"}
    SKIP = {"script", "style", "noscript", "template", "nav"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip_depth = max(0, self.skip_depth - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _HTMLText()
    parser.feed(html)
    parser.close()
    return "".join(parser.parts)


def pdf_to_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # optional dependency
        raise IngestError(f"{path.name}: PDF parsing needs `pip install pypdf`") from exc
    pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
    return "\n\n".join(pages)


def parse_file(path: Path) -> str:
    """Return the raw text of one document, chosen by file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return pdf_to_text(path)
    raw = path.read_text(encoding="utf-8")
    if suffix in (".md", ".markdown"):
        return markdown_to_text(raw)
    if suffix in (".html", ".htm"):
        return html_to_text(raw)
    if suffix == ".txt":
        return raw
    raise IngestError(f"{path.name}: unsupported file type {suffix!r}")


# ---------------------------------------------------------- preprocessing

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    """Normalise Unicode and whitespace while keeping paragraph boundaries."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)           # "re-\nturn" -> "return"
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        paragraph = " ".join(block.split())
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def _fingerprint(paragraph: str) -> str:
    return hashlib.sha1(" ".join(paragraph.lower().split()).encode()).hexdigest()


def drop_shared_boilerplate(documents: dict[str, str], min_documents: int = 2) -> dict[str, str]:
    """Remove paragraphs repeated verbatim across documents (footers, disclaimers).

    Within one document a repeated paragraph is kept once.
    """
    seen_in: dict[str, set[str]] = {}
    for name, text in documents.items():
        for paragraph in text.split("\n\n"):
            seen_in.setdefault(_fingerprint(paragraph), set()).add(name)
    cleaned = {}
    for name, text in documents.items():
        kept, local = [], set()
        for paragraph in text.split("\n\n"):
            key = _fingerprint(paragraph)
            if len(seen_in[key]) >= min_documents or key in local:
                continue
            local.add(key)
            kept.append(paragraph)
        cleaned[name] = "\n\n".join(kept)
    return cleaned


# ---------------------------------------------------------------- loading

@dataclass
class IngestReport:
    loaded: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    duplicates: list[str] = field(default_factory=list)


def load_documents(directory: Path, report: IngestReport | None = None) -> dict[str, str]:
    """Parse and clean every supported file in `directory`.

    Unparseable or empty files and byte-identical duplicates are skipped and
    recorded in `report`, so ingestion problems are visible instead of silent.
    """
    report = report if report is not None else IngestReport()
    documents, hashes = {}, {}
    for path in sorted(p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED):
        try:
            text = clean_text(parse_file(path))
        except (IngestError, UnicodeDecodeError) as exc:
            report.skipped[path.name] = str(exc)
            continue
        if not text:
            report.skipped[path.name] = "no extractable text"
            continue
        digest = _fingerprint(text)
        if digest in hashes:
            report.duplicates.append(f"{path.name} == {hashes[digest]}")
            continue
        hashes[digest] = path.name
        documents[path.name] = text
        report.loaded.append(path.name)
    return drop_shared_boilerplate(documents)
