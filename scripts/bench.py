"""
Quick benchmark: run N queries, collect retrieval + timing metrics, print summary.

Usage:
    python -m scripts.bench
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("PRESCISE_ENABLE_FORMULA_ENRICHMENT", "0")
os.environ.setdefault("PRESCISE_ENABLE_CODE_ENRICHMENT", "0")
os.environ.setdefault("PRESCISE_ENABLE_VLM_FALLBACK", "0")
os.environ.setdefault("PRESCISE_ENABLE_VLM_BACKGROUND_SCAN", "0")  # off during bench
os.environ.setdefault("PRESCISE_FORMULA_QUALITY_MODE", "balanced")

QUESTIONS = [
    "What are the Nosé-Hoover equations of motion for the thermostat?",
    "What is the canonical partition function and how is it related to temperature?",
    "How does the Verlet integration algorithm work for molecular dynamics?",
    "What is the Lennard-Jones potential and what are its parameters?",
    "How do Ewald summation methods handle long-range electrostatic interactions?",
    "What is ergodicity and why does it matter for molecular dynamics simulations?",
    "How do you set up a LAMMPS input script for a basic NVT simulation?",
    "What is the equipartition theorem and how is it used to measure temperature?",
]

TOP_K = 10


def classify_hit(item: dict) -> str:
    chunk = item["chunk"]
    if chunk.get("chunk_type") == "formula":
        return "formula"
    src = item.get("source", "")
    if "formula" in src:
        return "formula"
    return "text"


def run_bench():
    from core.persistence import IndexManager
    from core.embeddings import Embedder
    from core.formula.normalizer import normalize_formula
    from core.nlp.tokenizer import tokenize
    from core.llm import GeminiClient
    from core.agent.dd_expert_agent import DDExpertAgent
    from core.agent.context_builder import build_context_description

    print("Loading index …")
    t0 = time.time()
    mgr = IndexManager(pdf_dir="data/pdfs", index_dir="data/index")
    chunks_path = Path("data/index/chunks.pkl")
    retriever = (
        mgr._build_from_scratch()
        if not chunks_path.exists()
        else mgr.load_or_build()
    )
    embedder = Embedder()
    llm = GeminiClient()
    context_desc = build_context_description(getattr(retriever, "chunks", []))
    agent = DDExpertAgent(retriever, embedder, llm, context_desc)
    load_time = time.time() - t0
    print(f"Index loaded in {load_time:.1f}s\n")
    print(f"Text chunks : {len(retriever.chunks)}")
    print(f"Formula chunks: {len(retriever.formula_chunks)}\n")

    rows = []

    for qi, question in enumerate(QUESTIONS, 1):
        print(f"[{qi}/{len(QUESTIONS)}] {question[:70]}")
        t_start = time.time()

        # Retrieval
        tokens = tokenize(normalize_formula(question))
        emb = embedder.embed_text(question)
        ret_result = retriever.retrieve_with_router(
            query_tokens=tokens,
            query_embedding=emb,
            query_str=question,
            top_k=TOP_K,
        )
        t_retrieval = time.time() - t_start

        hits = ret_result["results"]
        router = ret_result["router_decision"]
        n_formula = sum(1 for h in hits if classify_hit(h) == "formula")
        n_text = len(hits) - n_formula
        top_scores = [round(h["score"], 4) for h in hits[:3]]

        # LLM answer
        t_llm_start = time.time()
        dd_result = agent.run(question)
        t_llm = time.time() - t_llm_start

        answer = dd_result.get("answer", "")
        answer_words = len(answer.split())

        t_total = time.time() - t_start

        row = {
            "q": qi,
            "question": question,
            "intent": router["intent"],
            "bm25_w": router["bm25_weight"],
            "faiss_w": router["faiss_weight"],
            "n_formula_hits": n_formula,
            "n_text_hits": n_text,
            "top3_scores": top_scores,
            "t_retrieval_s": round(t_retrieval, 2),
            "t_llm_s": round(t_llm, 2),
            "t_total_s": round(t_total, 2),
            "answer_words": answer_words,
            "answer_preview": answer[:200].replace("\n", " "),
        }
        rows.append(row)

        print(
            f"  intent={router['intent']:12s}  bm25={router['bm25_weight']:.2f}  "
            f"faiss={router['faiss_weight']:.2f}  "
            f"formula_hits={n_formula}/{TOP_K}  "
            f"retrieval={t_retrieval:.2f}s  llm={t_llm:.2f}s  total={t_total:.2f}s  "
            f"answer_words={answer_words}"
        )
        print(f"  preview: {answer[:120].replace(chr(10), ' ')}\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    def avg(key):
        return sum(r[key] for r in rows) / len(rows)

    print(f"Questions run         : {len(rows)}")
    print(f"Avg retrieval time    : {avg('t_retrieval_s'):.2f}s")
    print(f"Avg LLM time          : {avg('t_llm_s'):.2f}s")
    print(f"Avg total time/query  : {avg('t_total_s'):.2f}s")
    print(f"Avg formula hits (/{TOP_K})  : {avg('n_formula_hits'):.1f}")
    print(f"Avg text hits (/{TOP_K})     : {avg('n_text_hits'):.1f}")
    print(f"Avg answer length     : {avg('answer_words'):.0f} words")

    intents = {}
    for r in rows:
        intents[r["intent"]] = intents.get(r["intent"], 0) + 1
    print(f"Router intent dist    : {dict(sorted(intents.items(), key=lambda x: -x[1]))}")

    print("\nPer-question breakdown:")
    print(f"{'Q':>2}  {'Intent':12s}  {'BM25':5s}  {'FAISS':5s}  {'FmHits':7s}  "
          f"{'Retr(s)':7s}  {'LLM(s)':6s}  {'Tot(s)':6s}  {'Words':5s}")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['q']:>2}  {r['intent']:12s}  {r['bm25_w']:.2f}   "
            f"{r['faiss_w']:.2f}   {r['n_formula_hits']:>2}/{TOP_K}      "
            f"{r['t_retrieval_s']:>5.2f}s  {r['t_llm_s']:>5.2f}s  "
            f"{r['t_total_s']:>5.2f}s  {r['answer_words']:>5}"
        )


if __name__ == "__main__":
    run_bench()
