import unittest

from app.core import Passage, SearchIndex, extractive_answer
from app.evaluate import evaluate, groundedness

RETURNS = Passage("returns.md#1", "returns.md",
                  "Unopened items can be returned within 14 days of delivery with a receipt.", 0.5)


class GroundednessTests(unittest.TestCase):
    def test_extractive_answer_is_fully_grounded(self):
        result = groundedness(extractive_answer([RETURNS]), [RETURNS])
        self.assertEqual(result["score"], 1.0)

    def test_cited_but_wrong_number_is_flagged(self):
        answer = "Unopened items can be returned within 30 days of purchase without a receipt [returns.md#1]."
        result = groundedness(answer, [RETURNS])
        self.assertLess(result["score"], 0.95)
        self.assertEqual(len(result["unsupported"]), 1)

    def test_uncited_claim_is_unsupported(self):
        result = groundedness("Express shipping is free for members.", [RETURNS])
        self.assertEqual(result["score"], 0.0)

    def test_citation_to_unretrieved_passage_zeroes_the_score(self):
        answer = "Unopened items can be returned within 14 days [shipping.md#9]."
        result = groundedness(answer, [RETURNS])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["invalid_citations"], ["shipping.md#9"])

    def test_supported_and_unsupported_sentences_are_scored_separately(self):
        answer = ("Unopened items can be returned within 14 days with a receipt [returns.md#1]. "
                  "Gift wrapping is always free.")
        self.assertEqual(groundedness(answer, [RETURNS])["score"], 0.5)


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.index = SearchIndex({"returns.md": "Unopened items can be returned in 14 days with a receipt.",
                                  "shipping.md": "Shipping to other provinces takes 3 to 7 business days."})

    def answer(self, question):
        passages = self.index.search(question)
        return extractive_answer(passages), passages

    def test_metrics_cover_retrieval_abstention_facts_and_groundedness(self):
        cases = [{"question": "How many days to return an unopened item?", "source": "returns.md", "facts": ["14 days"]},
                 {"question": "What is the laptop warranty?", "source": None}]
        result = evaluate(self.answer, cases)
        self.assertEqual(result["metrics"], {"hit_rate": 1.0, "mrr": 1.0, "abstention_rate": 1.0,
                                             "fact_recall": 1.0, "groundedness": 1.0})
        self.assertEqual(result["failed"], {})

    def test_missing_fact_and_wrong_source_are_reported(self):
        cases = [{"question": "How many days to return an unopened item?", "source": "warranty.md", "facts": ["30 days"]}]
        result = evaluate(self.answer, cases)
        self.assertEqual(result["metrics"]["hit_rate"], 0.0)
        self.assertEqual(result["metrics"]["fact_recall"], 0.0)
        self.assertIn("hit_rate", result["failed"])
        self.assertFalse(result["results"][0]["pass"])


if __name__ == "__main__":
    unittest.main()
