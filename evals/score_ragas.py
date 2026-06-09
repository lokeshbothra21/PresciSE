"""
RAGAS eval — STEP 2 (runs in the ISOLATED ragas venv: .venv-ragas).

Reads the predictions file from STEP 1 and scores it with RAGAS, using Gemini as
the judge LLM + embeddings. Run with the eval venv's python:

    .venv-ragas/bin/python evals/score_ragas.py --predictions evals/predictions.jsonl

Metrics:
  - faithfulness        answer grounded in retrieved contexts (anti-hallucination)
  - answer_relevancy    answer addresses the question
  - context_precision   retrieved contexts are relevant / well-ranked
  - context_recall      retrieval got all needed info        (needs ground_truth)
  - answer_correctness  answer matches the reference         (needs ground_truth)
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="evals/predictions.jsonl")
    ap.add_argument("--out", default="evals/ragas_scores.csv")
    ap.add_argument("--model", default=os.getenv("PRESCISE_EVAL_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--embed-model", default=os.getenv("PRESCISE_EVAL_EMBED_MODEL", "models/gemini-embedding-001"))
    ap.add_argument("--workers", type=int, default=4, help="Lower this if you hit Gemini rate limits.")
    args = ap.parse_args()

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) — it's the judge LLM + embeddings.")

    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    with open(args.predictions, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        sys.exit(f"No rows in {args.predictions}")

    has_ref = all(r.get("ground_truth") for r in rows)
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r.get("contexts", []),
            reference=(r.get("ground_truth") or None),
        )
        for r in rows
    ]
    dataset = EvaluationDataset(samples=samples)

    llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model=args.model, google_api_key=key, temperature=0)
    )
    emb = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model=args.embed_model, google_api_key=key)
    )

    metrics = [faithfulness, answer_relevancy, context_precision]
    if has_ref:
        metrics += [context_recall, answer_correctness]
    else:
        print("[note] not every row has ground_truth → skipping context_recall + answer_correctness")

    result = evaluate(
        dataset, metrics=metrics, llm=llm, embeddings=emb,
        run_config=RunConfig(max_workers=args.workers),
    )
    print("\n=== RAGAS scores (averaged) ===")
    print(result)

    df = result.to_pandas()
    df.to_csv(args.out, index=False)
    print(f"\nPer-question scores → {args.out}")


if __name__ == "__main__":
    main()
