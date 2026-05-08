import os
import sys
import argparse
from dotenv import load_dotenv
from loguru import logger

from core.persistence import IndexManager
from core.agent import ScientificAnswerAgent
from core.llm import GeminiClient
from core.nlp.tokenizer import tokenize
from core.embeddings import Embedder
from core.utils import (
    print_retrieved_evidence, 
    print_section_header, 
    print_final_answer,
    format_clean_evidence
)


def main():
    parser = argparse.ArgumentParser(description="PresciSE PDF Demo")
    parser.add_argument("query", nargs="?", default="How does MD help in Drug Discovery?", help="The query to answer")
    parser.add_argument("--clean", action="store_true", help="Show only retrieved chunks and final answer")
    args = parser.parse_args()

    # Configure logging
    if args.clean:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    else:
        print("=== PRESCISE PDF DEMO ===\n")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not args.clean:
        print("API Key loaded?", bool(api_key))
        print_section_header("LOADING OR BUILDING INDEXES")
    
    # Use IndexManager for persistent indexes with incremental updates
    index_manager = IndexManager(pdf_dir="data/pdfs", index_dir="data/index")
    retriever = index_manager.load_or_build()
    
    if not args.clean:
        print_section_header("SETTING UP AI AGENT")

    # Agent now uses LangChain/LangGraph internally (maintains same interface)
    llm = GeminiClient()
    agent = ScientificAnswerAgent(llm)
    
    if not args.clean:
        print("  Agent ready (LangGraph workflow)\n")


    # Query
    query = args.query
    
    if not args.clean:
        print_section_header(f"QUERY: {query}")

    # Retrieve with router
    query_tokens = tokenize(query)
    embedder = Embedder()
    query_emb = embedder.embed_text(query)

    # Use router-based retrieval (dynamic weights)
    result = retriever.retrieve_with_router(
        query_tokens=query_tokens,
        query_embedding=query_emb,
        query_str=query,  # Original query for router
        top_k=15
    )
    
    # Extract results and router decision
    retrieved = result["results"]
    router_decision = result["router_decision"]
    
    # Display router decision
    if not args.clean:
        print("\n" + "="*60)
        print("ROUTER DECISION")
        print("="*60)
        print(f"  Intent:       {router_decision['intent']}")
        print(f"  BM25 Weight:  {router_decision['bm25_weight']:.3f}")
        print(f"  FAISS Weight: {router_decision['faiss_weight']:.3f}")
        print(f"  Reason:       {router_decision['reason']}")
        print("="*60 + "\n")

    # Print evidence
    if args.clean:
        format_clean_evidence(retrieved, top_k=10)
    else:
        print_retrieved_evidence(retrieved, top_k=15)

    # Generate answer
    if not args.clean:
        print_section_header("GENERATING ANSWER...")

    answer = agent.answer(query, retrieved)

    # Print final answer
    if args.clean:
        print("\n" + "="*60)
        print("FINAL ANSWER")
        print("="*60 + "\n")
        print(answer)
        print("\n" + "="*60 + "\n")
    else:
        print_final_answer(answer)


if __name__ == "__main__":
    main()
