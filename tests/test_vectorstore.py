from importlib.util import find_spec
import tempfile
import unittest

from app.core import SearchIndex

DOCS = {"returns.md": "Unopened items can be returned in 14 days with a receipt.",
        "shipping.md": "Shipping to other provinces takes 3 to 7 business days."}


@unittest.skipUnless(find_spec("qdrant_client"), "qdrant-client not installed")
class QdrantIndexTests(unittest.TestCase):
    def setUp(self):
        from app.vectorstore import QdrantIndex
        self.QdrantIndex = QdrantIndex

    def test_sparse_tfidf_in_qdrant_matches_in_memory_ranking_and_scores(self):
        question = "How many days to return an unopened item?"
        expected = SearchIndex(DOCS).search(question)
        found = self.QdrantIndex(DOCS).search(question)
        self.assertEqual([p.source_id for p in found], [p.source_id for p in expected])
        self.assertAlmostEqual(found[0].score, expected[0].score, places=3)

    def test_unknown_question_abstains(self):
        self.assertEqual(self.QdrantIndex(DOCS).search("What is the laptop warranty?"), [])

    def test_on_disk_collection_is_rebuilt_from_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.QdrantIndex(DOCS, path=tmp)
            first.client.close()
            second = self.QdrantIndex({"shipping.md": DOCS["shipping.md"]}, path=tmp)
            self.assertEqual(second.client.count(second.collection).count, 1)
            self.assertEqual(second.search("How long does shipping to other provinces take?")[0].filename,
                             "shipping.md")
            second.client.close()


if __name__ == "__main__":
    unittest.main()
