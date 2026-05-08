import os
import sys
from dotenv import load_dotenv

from core.persistence import IndexManager
from core.agent import ScientificAnswerAgent
from core.agent.query_enhancer import enhance_query
from core.llm import GeminiClient
from core.nlp.tokenizer import tokenize
from core.embeddings import Embedder
from core.utils import print_section_header


def main():
    print("=== PRESCISE ENHANCED QUERY DEMO ===\n")

    load_dotenv()
    print("GEMINI_API_KEY loaded?", bool(os.getenv("GEMINI_API_KEY")))
    
    # Setup
    print_section_header("LOADING OR BUILDING INDEXES")
    index_manager = IndexManager(pdf_dir="data/pdfs", index_dir="data/index")
    retriever = index_manager.load_or_build()
    
    print_section_header("SETTING UP AI AGENT")
    llm = GeminiClient()
    agent = ScientificAnswerAgent(llm)
    print("  Agent ready (LangGraph workflow)\n")

    # Query
    query = sys.argv[1] if len(sys.argv) > 1 else "How does MD help in Drug Discovery?"
    
    print_section_header(f"QUERY: {query}")
    
    # STEP 1: Query Enhancement
    print("\n" + "="*80)
    print("STEP 1: QUERY ENHANCEMENT")
    print("="*80 + "\n")
    
    enhancement = enhance_query(query)
    
    print(f"📊 Query Type: {enhancement['type'].upper()}")
    print(f"\n📝 Main Query: {enhancement['main_query']}")
    
    if enhancement['subqueries']:
        print(f"\n🔍 Generated Subqueries ({len(enhancement['subqueries'])} total):")
        for i, sq in enumerate(enhancement['subqueries'], 1):
            print(f"   {i}. {sq}")
    else:
        print("\n🔍 No subqueries generated")
    
    print(f"\n⚖️  Weights:")
    for key, weight in enhancement['weights'].items():
        print(f"   - {key}: {weight}")
    
    # STEP 2: Retrieval (for now, just use main query)
    print("\n" + "="*80)
    print("STEP 2: RETRIEVAL")
    print("="*80 + "\n")
    
    query_tokens = tokenize(query)
    embedder = Embedder()
    query_emb = embedder.embed_text(query)

    result = retriever.retrieve_with_router(
        query_tokens=query_tokens,
        query_embedding=query_emb,
        query_str=query,
        top_k=15
    )
    
    retrieved = result["results"]
    router_decision = result["router_decision"]
    
    print(f"  Intent:       {router_decision['intent']}")
    print(f"  BM25 Weight:  {router_decision['bm25_weight']:.3f}")
    print(f"  FAISS Weight: {router_decision['faiss_weight']:.3f}")
    print(f"  Retrieved:    {len(retrieved)} chunks")

    # STEP 3: Answer Generation
    print("\n" + "="*80)
    print("STEP 3: GENERATING ANSWER")
    print("="*80 + "\n")

    answer = agent.answer(query, retrieved)
    
    # STEP 4: Final Output
    print("\n" + "="*80)
    print("FINAL ANSWER")
    print("="*80 + "\n")
    print(answer)
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
