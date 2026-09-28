"""Reproducible retrieval evaluation on fictional, held-out questions.

No LLM is involved. Questions are separate from the indexed documents and are
intended to expose retrieval regressions, not to claim production accuracy.
"""
import json
from pathlib import Path
from .core import SearchIndex

ROOT = Path(__file__).resolve().parents[1]


def evaluate(index: SearchIndex, cases: list[dict]) -> dict:
    hits, abstentions = 0, 0
    results = []
    for case in cases:
        found = index.search(case["question"], top_k=3)
        expected = case["source"]
        success = (any(p.filename == expected for p in found) if expected else not found)
        hits += bool(success)
        abstentions += expected is None and not found
        results.append({"question": case["question"], "expected": expected,
                        "found": [p.filename for p in found], "pass": bool(success)})
    return {"passed": hits, "total": len(cases), "unknown_abstained": abstentions,
            "results": results}


if __name__ == "__main__":
    cases = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    result = evaluate(SearchIndex.from_directory(ROOT / "data"), cases)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] == result["total"] else 1)
