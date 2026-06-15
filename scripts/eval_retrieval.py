"""Retrieval-only eval: MRR + hit@k over a labeled set (offline, no LLM judge).

Run this to tune embeddings / reranker / chunking objectively — it measures
whether the *right chunks* are retrieved, separately from answer quality.

Testset JSON: a list of items, each with a question and a relevance label:
  {"question": "What is the SEI layer?",
   "user_id": "evaluser",
   "relevant_doc_ids": ["doc_a"],                                   # optional
   "relevant_contains": ["solid electrolyte interphase"]}           # optional
(Provide doc ids, must-contain substrings, or both. A retrieved chunk counts as
a hit if its doc_id matches OR its text contains a substring.)

Usage (from repo root):
  python -m scripts.eval_retrieval --testset evals/retrieval.example.json --top-k 10

Honors PRESCISE_EMBED_MODEL / PRESCISE_ENABLE_RERANK / PRESCISE_RERANKER_MODEL —
so you can A/B a change by re-running with different env and comparing the table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True)
    ap.add_argument("--user", default="evaluser")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    from core import settings
    from core.persistence import db
    from core.persistence.index_manager import IndexManager
    from core.embeddings.embedder import Embedder
    from core.nlp.tokenizer import tokenize
    from core.retrieval.eval_metrics import aggregate, query_metrics, rank_of_first_relevant

    db.init_db()
    embedder = Embedder()
    retriever = IndexManager(settings.pdf_dir(), settings.index_dir()).load_only()
    if os.getenv("PRESCISE_ENABLE_RERANK", "1") == "1":
        try:
            from core.retrieval.reranker import Reranker

            retriever.reranker = Reranker()
            print("[info] reranker attached")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] reranker unavailable ({exc}); evaluating without it")

    with open(args.testset, encoding="utf-8") as f:
        testset = json.load(f)

    per_query = []
    for i, item in enumerate(testset, 1):
        q = item["question"]
        allowed = {item.get("user_id", args.user), "__shared__"}
        res = retriever.retrieve_with_router(
            query_tokens=tokenize(q),
            query_embedding=embedder.embed_text(q),
            query_str=q,
            top_k=args.top_k,
            allowed_owners=allowed,
        )
        rank = rank_of_first_relevant(res["results"], item)
        m = query_metrics(rank)
        per_query.append(m)
        print(f"[{i}/{len(testset)}] rank={rank if rank else '—'} mrr={m['mrr']:.3f}  {q[:60]}")

    print("\n=== Retrieval metrics (averaged over %d queries) ===" % len(per_query))
    for k, v in aggregate(per_query).items():
        print(f"  {k:<8} {v}")


if __name__ == "__main__":
    main()
