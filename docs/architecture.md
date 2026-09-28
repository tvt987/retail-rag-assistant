# Architecture and evaluation decisions

## Request path

1. Ingest approved Markdown files at startup and split them into stable, cited chunks.
2. Embed with TF-IDF by default, or an optional local sentence-transformers model.
3. Retrieve top-k passages and abstain if the retrieval score is below the selected backend's threshold.
4. Produce an extractive answer by default. An optional LLM sees only retrieved excerpts and must cite source IDs; a missing citation falls back to the extractive result.
5. Return scores and excerpts so a reviewer can inspect evidence.

The LLM can still generate a claim unsupported by a cited excerpt. Do not present citations alone as a faithfulness guarantee.

## Evaluation

`eval/questions.json` contains answerable and unanswerable questions. The evaluator checks whether the expected document occurs in the top three and whether the system abstains on unknowns. Eight synthetic questions are only a regression fixture. A serious evaluation would add independently labeled retail data, answer correctness, groundedness, retrieval recall at k, latency, cost, and multilingual cases.

## Design references

- [LangChain RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch) for the separation of indexing, retrieval and generation.
- [Ragas](https://github.com/vibrantlabsai/ragas) for separating retrieval quality from answer faithfulness. This project does not claim to implement Ragas metrics.
- [Langfuse](https://github.com/langfuse/langfuse) for the idea of tracing retrieval and generation separately; full observability is a future improvement.

All code in this repository is independently written for this demo.
