from importlib.util import find_spec
import unittest

from app.core import SearchIndex

DOCS = {"returns.md": "Unopened items can be returned in 14 days with a receipt.",
        "shipping.md": "Shipping to other provinces takes 3 to 7 business days."}
QUESTION = {"question": "How many days to return an unopened item?"}


@unittest.skipUnless(find_spec("langchain_core"), "langchain-core not installed")
class LangChainPipelineTests(unittest.TestCase):
    def setUp(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from app.langchain_rag import IndexRetriever, build_chain
        self.FakeListChatModel = FakeListChatModel
        self.build_chain = build_chain
        self.retriever = IndexRetriever(index=SearchIndex(DOCS))

    def test_retriever_returns_documents_with_citation_metadata(self):
        docs = self.retriever.invoke(QUESTION["question"])
        self.assertEqual(docs[0].metadata["source_id"], "returns.md#1")
        self.assertIn("14 days", docs[0].page_content)

    def test_without_a_model_the_chain_returns_the_cited_extract(self):
        out = self.build_chain(self.retriever).invoke(QUESTION)
        self.assertEqual(out["mode"], "extractive")
        self.assertIn("[returns.md#1]", out["answer"])

    def test_cited_model_answer_is_returned(self):
        llm = self.FakeListChatModel(responses=["Within 14 days with a receipt [returns.md#1]."])
        out = self.build_chain(self.retriever, llm).invoke(QUESTION)
        self.assertEqual((out["mode"], out["answer"]), ("llm", "Within 14 days with a receipt [returns.md#1]."))

    def test_uncited_model_answer_falls_back_to_extract(self):
        llm = self.FakeListChatModel(responses=["You have 30 days."])
        out = self.build_chain(self.retriever, llm).invoke(QUESTION)
        self.assertEqual(out["mode"], "extractive")
        self.assertIn("[returns.md#1]", out["answer"])

    def test_model_failure_falls_back_to_extract(self):
        from langchain_core.runnables import RunnableLambda

        def boom(_):
            raise TimeoutError("provider down")
        out = self.build_chain(self.retriever, RunnableLambda(boom)).invoke(QUESTION)
        self.assertEqual(out["mode"], "extractive")

    def test_no_evidence_means_no_model_call(self):
        llm = self.FakeListChatModel(responses=["Laptops have a 2-year warranty [returns.md#1]."])
        out = self.build_chain(self.retriever, llm).invoke({"question": "What is the laptop warranty?"})
        self.assertEqual(out["mode"], "extractive")
        self.assertIn("could not find", out["answer"])
        self.assertEqual(out["sources"], [])


if __name__ == "__main__":
    unittest.main()
