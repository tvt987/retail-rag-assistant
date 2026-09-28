"""Dependency-free retrieval and grounded answer construction."""
from collections import Counter
from dataclasses import dataclass
from math import log, sqrt
from pathlib import Path
import re

WORDS = re.compile(r"[\w]+", re.UNICODE)
STOP = {"a", "an", "the", "is", "are", "can", "i", "my", "to", "for", "in", "of", "with", "what", "how", "many", "after", "do", "does", "and", "be"}


def tokens(text: str) -> list[str]:
    result = []
    for word in WORDS.findall(text.lower()):
        if word in STOP:
            continue
        if word in {"returned", "returns", "returning"}:
            word = "return"
        if word in {"items"}:
            word = "item"
        if word in {"days"}:
            word = "day"
        result.append(word)
    return result


def chunk_text(text: str, max_words: int = 95) -> list[str]:
    """Keep paragraphs together when possible; split unusually long paragraphs."""
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], []
    for paragraph in paragraphs:
        words = paragraph.split()
        if current and len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            current = []
        for start in range(0, len(words), max_words):
            part = words[start:start + max_words]
            if len(current) + len(part) > max_words:
                chunks.append(" ".join(current))
                current = []
            current.extend(part)
    if current:
        chunks.append(" ".join(current))
    return chunks


@dataclass(frozen=True)
class Passage:
    source_id: str
    filename: str
    text: str
    score: float


class SearchIndex:
    def __init__(self, documents: dict[str, str]):
        self.chunks = [(f"{name}#{i + 1}", name, part)
                       for name, text in sorted(documents.items())
                       for i, part in enumerate(chunk_text(text))]
        counts = [Counter(tokens(part)) for _, _, part in self.chunks]
        n = max(1, len(counts))
        df = Counter(word for count in counts for word in count)
        self.idf = {word: log((n + 1) / (frequency + 1)) + 1 for word, frequency in df.items()}
        self.vectors = [self._vector(count) for count in counts]

    @classmethod
    def from_directory(cls, directory: Path) -> "SearchIndex":
        return cls({p.name: p.read_text(encoding="utf-8") for p in directory.glob("*.md")})

    def _vector(self, count: Counter) -> dict[str, float]:
        weighted = {word: (1 + log(freq)) * self.idf[word]
                    for word, freq in count.items() if word in self.idf}
        norm = sqrt(sum(v * v for v in weighted.values())) or 1
        return {word: value / norm for word, value in weighted.items()}

    def search(self, question: str, top_k: int = 3, min_score: float = 0.08) -> list[Passage]:
        query = self._vector(Counter(tokens(question)))
        ranked = sorted(((sum(value * vector.get(word, 0) for word, value in query.items()), chunk)
                         for chunk, vector in zip(self.chunks, self.vectors)), reverse=True)
        return [Passage(source_id, name, text, round(score, 4))
                for score, (source_id, name, text) in ranked[:top_k] if score >= min_score]


class SemanticIndex(SearchIndex):
    """Optional neural embeddings, with the same cited Passage interface."""
    def __init__(self, documents: dict[str, str], model_name: str):
        super().__init__(documents)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode([text for _, _, text in self.chunks], normalize_embeddings=True)

    def search(self, question: str, top_k: int = 3, min_score: float = 0.3) -> list[Passage]:
        query = self.model.encode([question], normalize_embeddings=True)[0]
        ranked = sorted(((float(vector @ query), chunk) for chunk, vector in zip(self.chunks, self.embeddings)), reverse=True)
        return [Passage(source_id, name, text, round(score, 4))
                for score, (source_id, name, text) in ranked[:top_k] if score >= min_score]


def extractive_answer(passages: list[Passage]) -> str:
    if not passages:
        return "I could not find an answer in the available documents."
    return "\n\n".join(f"[{p.source_id}] {p.text}" for p in passages)
