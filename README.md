# Retail RAG Assistant

A small, inspectable retrieval-augmented assistant for retail policy questions. It parses and cleans source documents (Markdown, text, HTML, PDF), chunks them, retrieves relevant chunks from an in-memory index or a **Qdrant** vector database, and returns answers with citations, either through its own minimal pipeline or a **LangChain** (LCEL) chain. Answers are evaluated for retrieval quality, factual recall and **groundedness**.

The default TF-IDF, extractive mode works offline with the standard library and FastAPI only, with no API key and no model download.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api:app --reload
```

Open http://127.0.0.1:8000/docs. Try `POST /ask` with `{"question":"Can I return an unopened product after 5 days?"}`. Example policies in `data/` are fictional.

## Optional features

```bash
pip install -r requirements-extras.txt      # qdrant-client, langchain-core, pypdf, httpx
```

| Setting | Effect |
|---|---|
| `VECTOR_STORE=qdrant` | Store chunk vectors in Qdrant. Embedded in memory by default; `QDRANT_PATH=./qdrant_data` persists to disk; `QDRANT_URL=http://localhost:6333` uses a Qdrant server. Without an embedding model the TF-IDF vectors are stored as Qdrant sparse vectors, so this also works offline. |
| `RAG_PIPELINE=langchain` | Serve `/ask` through the LangChain chain: `retriever -> cited context -> prompt -> chat model -> citation check`. |
| `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Dense neural embeddings (needs `pip install sentence-transformers`), in memory or in Qdrant. |
| `OPENAI_API_KEY` (+ `OPENAI_MODEL`, `OPENAI_BASE_URL`) | Generate the answer with an OpenAI-compatible model. The LangChain pipeline also needs `pip install langchain-openai`. |

Both pipelines fail closed: with no evidence, a model error, or a generated answer that cites none of the retrieved passages, the response is the extractive, fully cited answer.

Container: `docker build -t retail-rag .` (add `--build-arg EXTRAS=true` for the optional features), then `docker run --rm -p 8000:8000 retail-rag`. Use `--env-file` for keys; never commit them.

## Ingestion

`app/ingest.py` turns the files in `data/` into clean text before indexing:

- **Parsing**: Markdown syntax is stripped (headings, links, emphasis, lists); HTML is read with the standard-library parser, dropping `script`, `style` and `nav`; PDF text is extracted with `pypdf`.
- **Preprocessing**: Unicode NFKC normalisation, control-character removal, re-joining words hyphenated across line breaks, whitespace collapse with paragraph boundaries kept, and removal of paragraphs repeated across documents (shared footers, disclaimers).
- **Reporting**: unsupported, empty and duplicate files are skipped and listed, and `/health` shows what was skipped.

## Evaluation

```bash
python -m unittest discover -s tests -v
python -m app.evaluate                                   # default pipeline, in-memory TF-IDF
python -m app.evaluate --store qdrant                    # Qdrant vector database
python -m app.evaluate --pipeline langchain --store qdrant
```

`eval/questions.json` holds 16 labelled questions across three documents (12 answerable, 4 unanswerable). Each answerable question lists the facts the answer must contain. The evaluator reports:

| Metric | Meaning |
|---|---|
| `hit_rate` | expected document is in the top 3 |
| `mrr` | mean reciprocal rank of the expected document |
| `abstention_rate` | unanswerable questions return no passages |
| `fact_recall` | expected facts appear in the answer |
| `groundedness` | share of answer sentences supported by a passage they cite: most content words and every number must appear in that passage, and citing a passage that was not retrieved scores zero |

CI runs the standard-library suite and the evaluation on every push, and a second job runs the Qdrant, LangChain, PDF and API tests plus both extra evaluations. All metrics are 1.0 on the current fixture.

Groundedness is a deterministic lexical check, not an LLM judge or a Ragas faithfulness score. In extractive mode it is 1.0 by construction; it exists to score generated answers, and `tests/test_evaluate.py` shows it rejecting a wrong quantity, an uncited claim and an invented citation. The questions are a small synthetic regression set; a production evaluation needs independently labelled, real queries.

## Design

`documents -> parse + clean -> paragraph-aware chunks -> TF-IDF / embeddings -> in-memory or Qdrant search -> cited excerpts -> optional LLM (direct or LangChain) -> citation check`

`GET /health`, `POST /ask` and `GET /documents` are the public endpoints. Retrieval is local and deterministic; no user question leaves the machine unless an API key is configured. Low similarity returns an explicit unknown answer. Responses include source IDs, filenames, passages and scores for inspection.

## Limits and next steps

The index (and the Qdrant collection) is rebuilt from `data/` on startup, so the store cannot drift from the approved documents, but large corpora would need incremental ingestion. Citations and the lexical groundedness check do not prove every generated claim; human verification is still needed. For production: document ACL filters, an LLM-judged faithfulness metric alongside the lexical one, a held-out real evaluation set, and tracing. Do not treat the fictional policy documents as real customer guidance.

See [architecture and evaluation decisions](docs/architecture.md) for design trade-offs and references.
