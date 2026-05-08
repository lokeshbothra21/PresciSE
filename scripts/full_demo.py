import os
print("SCRIPT SEES GEMINI_API_KEY?", bool(os.getenv("GEMINI_API_KEY")))

from dotenv import load_dotenv
load_dotenv()

from core.embeddings import Embedder
from core.llm import GeminiClient
from core.agent import ScientificAnswerAgent
from core.retrieval.hybrid_retriever import HybridRetriever
from core.nlp.tokenizer import tokenize


print("=== PRESCISE END-TO-END DEMO ===")

# Setup embedder + LLM + agent
embedder = Embedder()
llm = GeminiClient(model_name="gemini-2.5-flash")
agent = ScientificAnswerAgent(llm)

# Sample chunk corpus (later: replaced by real ingestion pipeline)
chunks = [
    {
        "chunk_id": "c1",
        "text": "Michaelis-Menten kinetics describes enzyme reaction rate behavior and saturation effects.",
        "tokens": tokenize("Michaelis-Menten kinetics describes enzyme reaction rate behavior and saturation effects."),
        "metadata": {"doc_id": "doc1"},
    },
    {
        "chunk_id": "c2",
        "text": "Allosteric regulation changes enzyme activity via conformational shifts and cooperative binding.",
        "tokens": tokenize("Allosteric regulation changes enzyme activity via conformational shifts and cooperative binding."),
        "metadata": {"doc_id": "doc2"},
    },
    {
        "chunk_id": "c3",
        "text": "Neural networks are used for computer vision and NLP tasks.",
        "tokens": tokenize("Neural networks are used for computer vision and NLP tasks."),
        "metadata": {"doc_id": "doc3"},
    },
]

# Add embeddings
texts = [c["text"] for c in chunks]
embs = embedder.embed_texts(texts)
for c, e in zip(chunks, embs):
    c["embedding"] = e

# Retriever
retriever = HybridRetriever(chunks, bm25_top_k=3, faiss_top_k=3)

# Query
query = "Explain Michaelis-Menten kinetics vs allosteric regulation"
query_tokens = tokenize(query)
query_emb = embedder.embed_text(query)

retrieved = retriever.retrieve(query_tokens=query_tokens, query_embedding=query_emb, top_k=2)

print("\nTop retrieved evidence:")
for item in retrieved:
    print("-", item["chunk"]["chunk_id"], "| score:", round(item["score"], 4))

# Agent answers
final_answer = agent.answer(query=query, retrieved_chunks=retrieved)

print("\n=== FINAL ANSWER ===")
print(final_answer)
