import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from core.agent.dd_expert_agent import DDExpertAgent
from core.agent.context_builder import build_context_description
from core.embeddings import Embedder
from core.formula.normalizer import normalize_formula
from core.llm import GeminiClient
from core.nlp.tokenizer import tokenize
from core.persistence import IndexManager
from core.utils.output_formatter import render_formula_answer


def _ensure_default_formula_runtime() -> None:
    """
    Use stable defaults for formula pipeline unless user overrides env manually.
    """
    os.environ.setdefault("PRESCISE_ENABLE_FORMULA_ENRICHMENT", "0")
    os.environ.setdefault("PRESCISE_ENABLE_CODE_ENRICHMENT", "0")
    os.environ.setdefault("PRESCISE_ENABLE_VLM_FALLBACK", "0")
    os.environ.setdefault("PRESCISE_FORMULA_QUALITY_MODE", "balanced")


def _run_nougat_scan(retriever, embedder) -> None:
    """
    Background worker — scans each unscanned PDF with Nougat once.

    Runs in a daemon thread so the CLI answer is returned immediately; the scan
    continues in the background and improves the in-memory formula index for any
    subsequent queries within the same process (or persists for future runs).
    """
    from core.formula.nougat_scanner import NougatFormulaScanner
    from core.persistence.index_manager import IndexManager
    from loguru import logger

    scanner = NougatFormulaScanner(embedder, "data/index")
    mgr = IndexManager("data/pdfs", "data/index")

    pdf_dir = Path("data/pdfs")
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        return

    total_added = 0
    total_t0 = time.time()

    for pdf_path in pdfs:
        doc_id = mgr._derive_doc_id(pdf_path)
        if not scanner.needs_scan(doc_id):
            continue

        prev_count = sum(
            1 for fc in retriever.formula_chunks
            if fc.get("doc_id") == doc_id
        )
        existing_doc = [
            fc for fc in retriever.formula_chunks if fc.get("doc_id") == doc_id
        ]
        others = [
            fc for fc in retriever.formula_chunks if fc.get("doc_id") != doc_id
        ]

        try:
            result = scanner.scan_pdf(str(pdf_path), doc_id, existing_doc)
        except Exception as exc:
            logger.error(f"[VLM SCAN] {doc_id}: scan failed: {exc}")
            continue

        new_all = others + result["merged"]
        retriever.update_formula_index(new_all)
        mgr.save_formula_chunks(new_all, retriever=retriever)

        scanner.mark_scanned(
            doc_id,
            result["pages_scanned"],
            result["added"],
            result["replaced"],
            result["elapsed_s"],
        )
        total_added += result["added"]

        logger.info(
            f"[VLM SCAN] {doc_id}: "
            f"{result['pages_scanned']} pages | "
            f"{result['elapsed_s']:.1f}s | "
            f"formulas: {prev_count} → {len(result['merged'])} "
            f"(+{result['added']} new, {result['replaced']} replaced)"
        )

    logger.info(
        f"[VLM SCAN] ✅ All docs scanned. "
        f"Total time: {time.time() - total_t0:.1f}s. "
        f"Total formulas added: {total_added}. "
        f"Index size: {len(retriever.formula_chunks)}"
    )


def _print_hits(results, top_k: int) -> None:
    print("\nTop hits:")
    for i, item in enumerate(results[:top_k], 1):
        chunk = item["chunk"]
        source = item.get("source", "text_index")
        score = item.get("score", 0.0)
        doc = chunk.get("doc_id", "unknown")

        if source == "formula_index":
            preview = chunk.get("normalized_formula") or chunk.get("formula_text") or ""
            kind = "FORMULA"
        else:
            preview = chunk.get("text", "")
            kind = "TEXT"

        preview = preview.replace("\n", " ").strip()
        if len(preview) > 180:
            preview = preview[:180] + "..."
        print(f"[{i}] {kind} | {source} | {score:.4f} | {doc}")
        print(f"    {preview}")


def main():
    parser = argparse.ArgumentParser(description="Short query runner for PresciSE")
    parser.add_argument("query", help="Question to ask")
    parser.add_argument("--top-k", type=int, default=10, help="How many hits to print")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild from scratch before query")
    args = parser.parse_args()

    load_dotenv()
    _ensure_default_formula_runtime()

    index_manager = IndexManager(pdf_dir="data/pdfs", index_dir="data/index")

    if args.rebuild:
        retriever = index_manager._build_from_scratch()
    else:
        # If no index yet, auto-build so command remains one-step.
        chunks_path = Path("data/index/chunks.pkl")
        retriever = index_manager._build_from_scratch() if not chunks_path.exists() else index_manager.load_or_build()

    embedder = Embedder()
    t0 = time.time()  # start of request timing (retrieval + agent)
    # Normalize query before tokenizing so Greek-letter names (zeta → ζ handled
    # in reverse by tokenizer) produce tokens that match the stored index tokens.
    query_tokens = tokenize(normalize_formula(args.query))
    query_emb = embedder.embed_text(args.query)

    result = retriever.retrieve_with_router(
        query_tokens=query_tokens,
        query_embedding=query_emb,
        query_str=args.query,
        top_k=max(args.top_k, 15),
    )
    retrieved = result["results"]
    router = result["router_decision"]

    print("\nRouter:")
    print(
        f"intent={router['intent']} bm25={router['bm25_weight']:.2f} "
        f"faiss={router['faiss_weight']:.2f} reason={router['reason']}"
    )

    _print_hits(retrieved, top_k=args.top_k)

    # Launch Nougat background scan (non-daemon — process stays alive until scan finishes)
    # Default is "0" in CLI; set PRESCISE_ENABLE_VLM_BACKGROUND_SCAN=1 to opt in.
    if os.getenv("PRESCISE_ENABLE_VLM_BACKGROUND_SCAN", "0") == "1":
        import threading
        t = threading.Thread(
            target=_run_nougat_scan, args=(retriever, embedder), daemon=False
        )
        t.start()

    llm = GeminiClient()
    context_desc = build_context_description(getattr(retriever, "chunks", []))
    agent = DDExpertAgent(retriever, embedder, llm, context_desc)
    llm.begin_request()  # reset per-request LLM stats
    dd_result = agent.run(args.query)
    print("\nAnswer:\n")
    print(render_formula_answer(dd_result["answer"]))

    stats = llm.request_stats()
    total_s = round(time.time() - t0, 2)
    print(
        f"\n[REQUEST STATS] llm_calls={stats['llm_calls']} | "
        f"llm_time={stats['llm_time_s']}s | total_time={total_s}s "
        f"(non-llm={round(total_s - stats['llm_time_s'], 2)}s)"
    )


if __name__ == "__main__":
    main()

