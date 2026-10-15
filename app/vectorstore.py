"""Qdrant-backed retrieval with the same cited Passage interface as SearchIndex.

Without an embedding model the TF-IDF vectors are stored as Qdrant *sparse*
vectors, so the vector database works offline with no model download. With
EMBEDDING_MODEL set, dense sentence-transformers vectors are stored instead.

Qdrant runs embedded (":memory:" or a local directory) or against a server via
`url`. Needs `pip install qdrant-client` (see requirements-extras.txt).
"""
from collections import Counter

from .core import Passage, SearchIndex, tokens


class QdrantIndex(SearchIndex):
    def __init__(self, documents: dict[str, str], *, path: str = ":memory:", url: str | None = None,
                 collection: str = "retail_policies", model_name: str | None = None):
        super().__init__(documents)  # chunks and TF-IDF weights
        from qdrant_client import QdrantClient, models

        self.models = models
        self.collection = collection
        if url:
            self.client = QdrantClient(url=url)
        elif path == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(path=path)

        self.model = None
        if model_name:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.backend = "qdrant-dense"
            self.default_min_score = 0.3
        else:
            self.vocabulary = {word: i for i, word in enumerate(sorted(self.idf))}
            self.backend = "qdrant-sparse-tfidf"
            self.default_min_score = 0.08
        self._rebuild_collection()

    # The collection is rebuilt from the source documents on every start, so
    # the store can never drift from the approved documents in data/.
    def _rebuild_collection(self) -> None:
        m = self.models
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        if self.model is not None:
            embeddings = self.model.encode([text for _, _, text in self.chunks], normalize_embeddings=True)
            self.client.create_collection(
                self.collection,
                vectors_config=m.VectorParams(size=len(embeddings[0]), distance=m.Distance.COSINE))
            vectors = [list(map(float, e)) for e in embeddings]
        else:
            self.client.create_collection(self.collection, vectors_config={},
                                          sparse_vectors_config={"tfidf": m.SparseVectorParams()})
            vectors = [{"tfidf": self._sparse(v)} for v in self.vectors]
        self.client.upsert(self.collection, points=[
            m.PointStruct(id=i, vector=vector,
                          payload={"source_id": source_id, "filename": name, "text": text})
            for i, ((source_id, name, text), vector) in enumerate(zip(self.chunks, vectors))])

    def _sparse(self, weights: dict[str, float]):
        pairs = sorted((self.vocabulary[w], v) for w, v in weights.items() if w in self.vocabulary)
        return self.models.SparseVector(indices=[i for i, _ in pairs], values=[v for _, v in pairs])

    def search(self, question: str, top_k: int = 3, min_score: float | None = None) -> list[Passage]:
        min_score = self.default_min_score if min_score is None else min_score
        if self.model is not None:
            query = list(map(float, self.model.encode([question], normalize_embeddings=True)[0]))
            response = self.client.query_points(self.collection, query=query, limit=top_k, with_payload=True)
        else:
            sparse = self._sparse(self._vector(Counter(tokens(question))))
            if not sparse.indices:
                return []
            response = self.client.query_points(self.collection, query=sparse, using="tfidf",
                                                limit=top_k, with_payload=True)
        return [Passage(p.payload["source_id"], p.payload["filename"], p.payload["text"], round(p.score, 4))
                for p in response.points if p.score >= min_score]
