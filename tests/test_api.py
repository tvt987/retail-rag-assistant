from importlib import import_module, reload
from importlib.util import find_spec
import os
import unittest
from unittest import mock

HAS_API = all(find_spec(m) for m in ("fastapi", "httpx"))


def client_for(**env):
    """Import app.api under the given environment (configuration is read at import)."""
    from fastapi.testclient import TestClient
    clean = {k: v for k, v in os.environ.items()
             if k not in ("OPENAI_API_KEY", "EMBEDDING_MODEL", "VECTOR_STORE", "RAG_PIPELINE")}
    with mock.patch.dict(os.environ, {**clean, **env}, clear=True):
        api = reload(import_module("app.api"))
    return TestClient(api.app)


@unittest.skipUnless(HAS_API, "fastapi/httpx not installed")
class ApiTests(unittest.TestCase):
    def test_health_reports_backend_and_ingestion(self):
        body = client_for().get("/health").json()
        self.assertEqual((body["retrieval"], body["pipeline"], body["documents_skipped"]), ("tfidf", "default", {}))
        self.assertEqual(client_for().get("/documents").json()["documents"],
                         ["returns.md", "shipping.md", "store_pickup.html"])

    def test_ask_returns_cited_answer_from_html_document(self):
        body = client_for().post("/ask", json={"question": "How long will the store hold my pickup order?"}).json()
        self.assertEqual(body["mode"], "extractive")
        self.assertEqual(body["sources"][0]["filename"], "store_pickup.html")
        self.assertIn("[store_pickup.html#1]", body["answer"])

    @unittest.skipUnless(find_spec("qdrant_client") and find_spec("langchain_core"), "extras not installed")
    def test_langchain_pipeline_over_qdrant_honours_top_k(self):
        client = client_for(VECTOR_STORE="qdrant", RAG_PIPELINE="langchain")
        self.assertEqual(client.get("/health").json()["retrieval"], "qdrant-sparse-tfidf")
        body = client.post("/ask", json={"question": "How long does a refund take after inspection?", "top_k": 1}).json()
        self.assertEqual(len(body["sources"]), 1)
        self.assertEqual(body["sources"][0]["filename"], "returns.md")


if __name__ == "__main__":
    unittest.main()
