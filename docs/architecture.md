# Architecture and evaluation decisions

## Request path

1. Parse approved Markdown, text, HTML and PDF files in `data/` (`app/ingest.py`), normalise the text and drop boilerplate shared across documents. Skipped files are reported, not silently ignored.
2. Split documents into stable, cited, paragraph-aware chunks.
3. Embed with TF-IDF by default, or an optional local sentence-transformers model.
4. Store and search vectors in memory, or in Qdrant (`app/vectorstore.py`). Without a neural model, TF-IDF vectors are stored as Qdrant sparse vectors, so the vector database needs no model download and returns the same scores as the in-memory index.
5. Retrieve top-k passages and abstain if the score is below the backend's threshold.
6. Produce an extractive answer by default. An optional LLM sees only retrieved excerpts and must cite source IDs; a missing citation or a model error falls back to the extractive result. The same rule is implemented twice: a minimal direct HTTP call (`app/api.py`) and a LangChain LCEL chain (`app/langchain_rag.py`), selectable with `RAG_PIPELINE`.
7. Return scores and excerpts so a reviewer can inspect evidence.

The LLM can still generate a claim unsupported by a cited excerpt. Do not present citations alone as a faithfulness guarantee.

## Why these choices

- **Rebuild the vector collection on startup.** The store can never contain chunks from documents that were removed or edited. For a large corpus this would become incremental upserts keyed by a content hash.
- **Sparse TF-IDF in Qdrant.** Keeps the vector-database path testable offline and in CI, and makes the in-memory and Qdrant backends directly comparable (`tests/test_vectorstore.py`).
- **LangChain as an alternative, not a requirement.** The default path stays dependency-free and easy to read; the LCEL chain shows the same retrieval and safety rules in the framework most teams use.

## Evaluation

`eval/questions.json` contains answerable questions with expected facts, and unanswerable questions. `app/evaluate.py` reports hit rate and MRR for retrieval, abstention on unknowns, fact recall, and a lexical groundedness score per answer sentence (content-word overlap with a cited passage, exact match for every number, and zero if a cited ID was not retrieved). The fixture is a regression check; a serious evaluation would add independently labelled retail data, an LLM-judged faithfulness metric, latency, cost and multilingual cases.

## Design references

- [LangChain RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch) for the separation of indexing, retrieval and generation.
- [Qdrant sparse vectors](https://qdrant.tech/documentation/concepts/vectors/#sparse-vectors) for storing lexical vectors in a vector database.
- [Ragas](https://github.com/vibrantlabsai/ragas) for separating retrieval quality from answer faithfulness. This project's groundedness check is a simpler lexical heuristic and does not implement Ragas metrics.
- [Langfuse](https://github.com/langfuse/langfuse) for the idea of tracing retrieval and generation separately; full observability is a future improvement.

All code in this repository is independently written for this demo.
