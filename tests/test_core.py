import unittest
from app.core import SearchIndex, chunk_text, extractive_answer


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = SearchIndex({"returns.md": "Unopened items can be returned in 14 days with a receipt.",
                                  "shipping.md": "Shipping to other provinces takes 3 to 7 business days."})

    def test_retrieves_relevant_source_with_citation(self):
        results = self.index.search("How many days to return an unopened item?")
        self.assertEqual(results[0].filename, "returns.md")
        self.assertIn("[returns.md#1]", extractive_answer(results))

    def test_unknown_question_does_not_fabricate_answer(self):
        self.assertEqual(self.index.search("What is the warranty for a laptop?"), [])
        self.assertIn("could not find", extractive_answer([]))

    def test_long_paragraph_splits_without_losing_words(self):
        text = " ".join(["product"] * 210)
        chunks = chunk_text(text, max_words=80)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(sum(len(x.split()) for x in chunks), 210)


if __name__ == "__main__":
    unittest.main()
