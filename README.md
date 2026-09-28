# Retail RAG Assistant

A small, inspectable retrieval-augmented assistant for retail policy questions. It chunks source documents, ranks relevant chunks, and returns answers with citations. It supports optional local neural embeddings and optional OpenAI-compatible answer generation. The default TF-IDF and extractive mode works offline without an API key.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api:app --reload
```

Open http://127.0.0.1:8000/docs. Try `POST /ask` with `{"question":"Can I return an unopened product after 5 days?"}`. Example policies in `data/` are fictional. No API key is needed.

Optional generation: set `OPENAI_API_KEY` and, if needed, `OPENAI_MODEL` (default `gpt-4o-mini`) and `OPENAI_BASE_URL` (default `https://api.openai.com/v1`). The retrieved passages and citation IDs are sent to the model. If the model is unavailable, the response fails closed to an extractive answer.

For semantic retrieval, install `pip install sentence-transformers` and set `EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. The model is downloaded on first run; the API starts after embeddings are built. Leave it unset for the lightweight offline TF-IDF baseline.

```bash
python -m unittest discover -s tests -v
python -m app.evaluate
```

Container: `docker build -t retail-rag . && docker run --rm -p 8000:8000 retail-rag`. Use environment variables with `--env-file` if opting into an external model; never commit keys.

## Design

`documents -> paragraph-aware chunks -> TF-IDF or neural embeddings -> cosine search -> cited excerpts -> optional LLM answer`

`GET /health`, `POST /ask`, and `GET /documents` are the public endpoints. Retrieval is local and deterministic; no user question is sent to a third party unless an API key is configured. Low similarity returns an explicit unknown answer. The API response includes source IDs, filenames, passages and scores for inspection.

`eval/questions.json` holds labeled queries, including unanswerable questions. `python -m app.evaluate` checks source retrieval in the top three and abstention on the unknowns; CI runs it after every change. This is a small synthetic evaluation and should be replaced with real, independently labeled queries for a production use case.

## Limits and next steps

The in-memory index rebuilds on startup. The optional LLM is instructed to ground answers in retrieved excerpts, but citations alone do not prove every generated claim; human verification is still needed. For production: use persistent vector storage, document ACL filters, a held-out evaluation set, tracing, and ingestion of approved documents. Do not treat the fictional policy documents as real customer guidance.

See [architecture and evaluation decisions](docs/architecture.md) for design trade-offs and references.
