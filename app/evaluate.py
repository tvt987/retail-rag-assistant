"""Reproducible RAG evaluation on fictional, held-out questions.

Retrieval  hit@k, MRR and abstention on unanswerable questions.
Answers    fact recall (expected facts appear in the answer) and groundedness.

Groundedness is a deterministic lexical check: every answer sentence must have
most of its content words, and every number, present in a passage it cites
(citations carry over to the following sentences, which matches the extractive
format), and every cited ID must be one of the retrieved passages. It catches
wrong quantities, uncited claims and invented citations cheaply. It is not an LLM-judge or Ragas
faithfulness score, and the small fictional question set is a regression
check, not a production accuracy claim.

    python -m app.evaluate                          # TF-IDF, default pipeline
    python -m app.evaluate --store qdrant           # Qdrant vector database
    python -m app.evaluate --pipeline langchain     # LangChain LCEL chain
"""
import argparse
import json
from pathlib import Path
import re
from typing import Callable

from .core import Passage, SearchIndex, extractive_answer, tokens
from .ingest import load_documents

ROOT = Path(__file__).resolve().parents[1]
CITATION = re.compile(r"\[([^\[\]\s]+#\d+)\]")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
SUPPORT_THRESHOLD = 0.6
REQUIRED = {"hit_rate": 1.0, "abstention_rate": 1.0, "fact_recall": 1.0, "groundedness": 0.95}

AnswerFn = Callable[[str], tuple[str, list[Passage]]]


def groundedness(answer: str, passages: list[Passage]) -> dict:
    by_id = {p.source_id: set(tokens(p.text)) for p in passages}
    claims, supported, unsupported, invalid = 0, 0, [], []
    current: list[str] = []
    for sentence in SENTENCE_END.split(answer):
        cited = CITATION.findall(sentence)
        if cited:
            current = cited
        invalid += [c for c in cited if c not in by_id]
        words = set(tokens(CITATION.sub(" ", sentence)))
        if not words:
            continue
        claims += 1
        evidence = [by_id[c] for c in current if c in by_id]
        numbers = {w for w in words if any(ch.isdigit() for ch in w)}
        # Most content words must overlap, and every number must appear verbatim:
        # a wrong quantity ("30 days" for "14 days") is the typical RAG error.
        best = max((len(words & e) / len(words) for e in evidence if numbers <= e), default=0.0)
        if best >= SUPPORT_THRESHOLD:
            supported += 1
        else:
            unsupported.append(sentence.strip())
    score = supported / claims if claims else 1.0
    if invalid:
        score = 0.0  # a citation to a passage that was never retrieved is a fabrication
    return {"score": round(score, 3), "claims": claims, "supported": supported,
            "unsupported": unsupported, "invalid_citations": sorted(set(invalid))}


def evaluate(answer_fn: AnswerFn, cases: list[dict]) -> dict:
    answerable = [c for c in cases if c["source"]]
    unknown = [c for c in cases if not c["source"]]
    hits = rr = facts_found = facts_total = abstained = 0
    grounded_scores, results = [], []
    for case in cases:
        answer, passages = answer_fn(case["question"])
        files = [p.filename for p in passages]
        row = {"question": case["question"], "expected": case["source"], "found": files}
        if case["source"]:
            rank = files.index(case["source"]) + 1 if case["source"] in files else None
            hits += rank is not None
            rr += 1 / rank if rank else 0
            facts = case.get("facts", [])
            present = [f for f in facts if f.lower() in answer.lower()]
            facts_found += len(present)
            facts_total += len(facts)
            grounded = groundedness(answer, passages)
            grounded_scores.append(grounded["score"])
            row.update(rank=rank, facts_missing=[f for f in facts if f not in present],
                       groundedness=grounded["score"], unsupported=grounded["unsupported"])
            row["pass"] = bool(rank) and len(present) == len(facts) and grounded["score"] >= REQUIRED["groundedness"]
        else:
            abstained += not passages
            row["pass"] = not passages
        results.append(row)

    metrics = {
        "hit_rate": round(hits / len(answerable), 3) if answerable else 1.0,
        "mrr": round(rr / len(answerable), 3) if answerable else 1.0,
        "abstention_rate": round(abstained / len(unknown), 3) if unknown else 1.0,
        "fact_recall": round(facts_found / facts_total, 3) if facts_total else 1.0,
        "groundedness": round(sum(grounded_scores) / len(grounded_scores), 3) if grounded_scores else 1.0,
    }
    failed = {k: v for k, v in metrics.items() if k in REQUIRED and v < REQUIRED[k]}
    return {"metrics": metrics, "required": REQUIRED, "failed": failed,
            "passed": sum(r["pass"] for r in results), "total": len(cases), "results": results}


def build_answer_fn(pipeline: str, store: str) -> AnswerFn:
    documents = load_documents(ROOT / "data")
    if store == "qdrant":
        from .vectorstore import QdrantIndex
        index = QdrantIndex(documents)
    else:
        index = SearchIndex(documents)

    if pipeline == "langchain":
        from .langchain_rag import IndexRetriever, build_chain, default_llm
        chain = build_chain(IndexRetriever(index=index), default_llm())

        def run_chain(question: str) -> tuple[str, list[Passage]]:
            out = chain.invoke({"question": question})
            return out["answer"], out["sources"]
        return run_chain

    def run_default(question: str) -> tuple[str, list[Passage]]:
        passages = index.search(question, top_k=3)
        return extractive_answer(passages), passages
    return run_default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pipeline", choices=["default", "langchain"], default="default")
    parser.add_argument("--store", choices=["memory", "qdrant"], default="memory")
    args = parser.parse_args(argv)
    cases = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    result = evaluate(build_answer_fn(args.pipeline, args.store), cases)
    print(json.dumps(result, indent=2))
    return 0 if not result["failed"] and result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
