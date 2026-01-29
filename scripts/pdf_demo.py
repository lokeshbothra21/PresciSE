import os
import sys
from glob import glob
from dotenv import load_dotenv

from core.ingestion.pdf_loader import load_pdf_as_document
from core.chunking.pdf_chunker import make_chunks_from_doc

from core.embeddings import Embedder
from core.retrieval.hybrid_retriever import HybridRetriever
from core.agent import ScientificAnswerAgent
from core.llm import GeminiClient
from core.nlp.tokenizer import tokenize


def main():
    print("=== PRESCISE PDF DEMO ===\n")

    load_dotenv()
    print("GEMINI_API_KEY loaded?", bool(os.getenv("GEMINI_API_KEY")))

    pdf_paths = glob("data/pdfs/*.pdf")
    if not pdf_paths:
        print("No PDFs found in data/pdfs/")
        return

    print(f"Found {len(pdf_paths)} PDF(s)\n")

    # 1) Ingest PDFs -> documents
    print("Step 1: Ingesting PDFs...")
    documents = []
    for p in pdf_paths:
        print(f"  - {os.path.basename(p)}")
        doc = load_pdf_as_document(p)
        documents.append(doc)
        print(f"    Loaded {len(doc.sections)} sections")

    # 2) Documents -> chunks
    print("\nStep 2: Chunking documents...")
    all_chunks = []
    for doc in documents:
        chunks = make_chunks_from_doc(doc, max_chars=1200)
        all_chunks.extend(chunks)
    print(f"  Created {len(all_chunks)} chunks")

    # 3) Embed chunks
    print("\nStep 3: Embedding chunks...")
    embedder = Embedder()
    texts = [c["text"] for c in all_chunks]
    embs = embedder.embed_texts(texts)
    
    for c, e in zip(all_chunks, embs):
        c["embedding"] = e
    print(f"  Embedded {len(embs)} chunks")

    # 4) Hybrid retrieval setup
    print("\nStep 4: Setting up hybrid retriever...")
    retriever = HybridRetriever(
        all_chunks,
        bm25_top_k=10,
        faiss_top_k=10,
        bm25_weight=0.6,
        faiss_weight=0.4
    )
    print("  Retriever ready")

    # 5) Agent setup
    print("\nStep 5: Initializing AI agent...")
    llm = GeminiClient(model_name="gemini-2.5-flash")
    agent = ScientificAnswerAgent(llm)
    print("  Agent ready")

    # 6) Query - accept from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "How does MD help in Drug Discovery?"
    
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}\n")
    
    query_tokens = tokenize(query)
    query_emb = embedder.embed_text(query)

    retrieved = retriever.retrieve(query_tokens=query_tokens, query_embedding=query_emb, top_k=6)

    print("Top retrieved evidence:")
    for i, item in enumerate(retrieved, 1):
        ch = item["chunk"]
        md = ch.get("metadata", {})
        print(f"  [{i}] Score: {item['score']:.4f} | Section: {md.get('section_type', 'N/A')} | Pages: {md.get('pages', 'N/A')}")
        print(f"      Preview: {ch['text'][:100]}...")

    # 7) Generate final answer
    print("\n" + "="*60)
    print("GENERATING ANSWER...")
    print("="*60 + "\n")
    
    final_answer = agent.answer(query=query, retrieved_chunks=retrieved)

    print("=== FINAL ANSWER ===")
    print(final_answer)
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
