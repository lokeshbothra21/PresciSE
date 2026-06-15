"""
RAGAS eval — STEP 1 (runs in the MAIN PresciSE venv).

Runs each test question through the real DDExpertAgent and records the answer +
the retrieved contexts, writing a predictions file that STEP 2 (score_ragas.py,
in the isolated ragas venv) consumes. Keeping generation and scoring in separate
venvs avoids the ragas↔langchain version conflict.

Usage (from the repo root):
    python evals/generate_predictions.py --testset evals/testset.example.json \
        --seed-pdf data/pdfs/<your>.pdf --user evaluser

  --seed-pdf is optional: it indexes that PDF for --user first, so you can do a
  full functional check even with a fresh/empty DB. Omit it to evaluate against
  documents already uploaded for --user.
"""

import argparse
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()


def _chunk_text(ch: dict) -> str:
    if ch.get("chunk_type") == "formula":
        return (ch.get("latex_formula") or ch.get("normalized_formula")
                or ch.get("formula_text") or ch.get("text", ""))
    return ch.get("text", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", required=True, help="JSON list of {question, ground_truth?, user_id?}")
    ap.add_argument("--out", default="evals/predictions.jsonl")
    ap.add_argument("--user", default="evaluser")
    ap.add_argument("--seed-pdf", default=None, help="Index this PDF for --user before evaluating.")
    ap.add_argument("--force-seed", action="store_true", help="Seed even if --user already has chunks (may duplicate).")
    ap.add_argument("--max-contexts", type=int, default=10)
    args = ap.parse_args()

    from core import settings
    from core.persistence import db
    from core.persistence.index_manager import IndexManager
    from core.embeddings.embedder import Embedder
    from core.llm.gemini_client import GeminiClient
    from core.agent.dd_expert_agent import DDExpertAgent
    from core.agent.context_builder import build_context_description
    from core.nlp.tokenizer import tokenize

    db.init_db()
    embedder = Embedder()
    mgr = IndexManager(settings.pdf_dir(), settings.index_dir())
    retriever = mgr.load_only()

    if args.seed_pdf:
        existing = db.count_chunks_for_owner(args.user)
        if existing and not args.force_seed:
            print(f"[seed] user '{args.user}' already has {existing} chunks — skipping seed "
                  f"(re-seeding would duplicate; pass --force-seed to override).")
        else:
            doc_id = "eval_" + uuid.uuid4().hex[:8]
            print(f"[seed] indexing {args.seed_pdf} as doc_id={doc_id} for user '{args.user}' …")
            mgr.index_uploaded_pdf(args.seed_pdf, doc_id, args.user, embedder, retriever)

    llm = GeminiClient()
    if getattr(llm, "is_mock", False):
        print("[WARN] GEMINI_API_KEY not set — answers will be MOCK. Set the key for a real eval.")
    ctx_desc = build_context_description(getattr(retriever, "chunks", []))
    agent = DDExpertAgent(retriever, embedder, llm, ctx_desc)

    with open(args.testset, encoding="utf-8") as f:
        testset = json.load(f)

    rows = []
    for i, item in enumerate(testset, 1):
        q = item["question"]
        gt = item.get("ground_truth", "")
        uid = item.get("user_id", args.user)
        allowed = {uid, "__shared__"}
        print(f"[{i}/{len(testset)}] {q[:70]}")

        # Answer comes from the full agent (its own internal retrieval).
        res = agent.run(q, allowed_owners=allowed)

        # Contexts for RAGAS = the chunks the AGENT actually retrieved while
        # answering (deduped, best-score first). Scoring the agent's real
        # evidence — not a separate query — keeps faithfulness and context_recall
        # valid. (Cross-subquestion scores aren't perfectly comparable, so
        # context_precision ORDER is approximate; the other metrics are unaffected.)
        best: dict = {}
        for it in res.get("all_retrieved_chunks", []):
            ch = it.get("chunk", {})
            cid = ch.get("chunk_id")
            if not cid:
                continue
            if cid not in best or float(it.get("score", 0.0)) > float(best[cid].get("score", 0.0)):
                best[cid] = it
        ctxs = []
        for it in sorted(best.values(), key=lambda x: -float(x.get("score", 0.0))):
            t = _chunk_text(it.get("chunk", {})).strip()
            if t:
                ctxs.append(t)
            if len(ctxs) >= args.max_contexts:
                break

        rows.append({"question": q, "answer": res["answer"], "contexts": ctxs, "ground_truth": gt})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} predictions → {args.out}")
    print("Next: score in the ragas venv →  .venv-ragas/bin/python evals/score_ragas.py")


if __name__ == "__main__":
    main()
