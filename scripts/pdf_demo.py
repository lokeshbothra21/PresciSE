import os
import sys
from dotenv import load_dotenv

from core.persistence import IndexManager
from core.agent import ScientificAnswerAgent
from core.llm import GeminiClient
from core.nlp.tokenizer import tokenize
from core.embeddings import Embedder
from core.utils import print_retrieved_evidence, print_section_header, print_final_answer


def main():
    print("=== PRESCISE PDF DEMO ===\n")

    load_dotenv()
    print("GEMINI_API_KEY loaded?", bool(os.getenv("GEMINI_API_KEY")))
    
    # Use IndexManager for persistent indexes with incremental updates
    print_section_header("LOADING OR BUILDING INDEXES")
    
    index_manager = IndexManager(pdf_dir="data/pdfs", index_dir="data/index")
    retriever = index_manager.load_or_build()
    
    print_section_header("SETTING UP AI AGENT")

    llm = GeminiClient()
    agent = ScientificAnswerAgent(llm)
    print("  Agent ready\n")


    # Query
    query = sys.argv[1] if len(sys.argv) > 1 else "How does MD help in Drug Discovery?"
    
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
    print("\n" + "="*60)
    print("ROUTER DECISION")
    print("="*60)
    print(f"  Intent:       {router_decision['intent']}")
    print(f"  BM25 Weight:  {router_decision['bm25_weight']:.3f}")
    print(f"  FAISS Weight: {router_decision['faiss_weight']:.3f}")
    print(f"  Reason:       {router_decision['reason']}")
    print("="*60 + "\n")

    # Print evidence using standard formatter
    print_retrieved_evidence(retrieved, top_k=15)

    # Generate answer
    print_section_header("GENERATING ANSWER...")

    answer = agent.answer(query, retrieved)

    # Print final answer using standard formatter
    print_final_answer(answer)


if __name__ == "__main__":
    main()
