from core.embeddings import Embedder
from core.retrieval.hybrid_retriever import HybridRetriever
from core.nlp.tokenizer import tokenize

print("=== PRESCISE HYBRID RETRIEVAL SANITY CHECK ===")

embedder = Embedder()

chunks = [
    {
        "chunk_id": "c1",
        "text": "Michaelis-Menten kinetics describes enzyme reaction rate behavior.",
        "tokens": tokenize("Michaelis-Menten kinetics describes enzyme reaction rate behavior."),
        "metadata": {"doc_id": "doc1"},
    },
    {
        "chunk_id": "c2",
        "text": "Allosteric regulation changes enzyme activity through conformational shifts.",
        "tokens": tokenize("Allosteric regulation changes enzyme activity through conformational shifts."),
        "metadata": {"doc_id": "doc2"},
    },
    {
        "chunk_id": "c3",
        "text": "Neural networks are used for computer vision and NLP tasks.",
        "tokens": tokenize("Neural networks are used for computer vision and NLP tasks."),
        "metadata": {"doc_id": "doc3"},
    },
]

# Attach embeddings to chunks
texts = [c["text"] for c in chunks]
embs = embedder.embed_texts(texts)
for c, e in zip(chunks, embs):
    c["embedding"] = e

retriever = HybridRetriever(
    chunks,
    bm25_top_k=3,
    faiss_top_k=3,
    bm25_weight=0.6,
    faiss_weight=0.4
)

query = "enzyme kinetics"
query_tokens = tokenize(query)
query_emb = embedder.embed_text(query)

results = retriever.retrieve(query_tokens=query_tokens, query_embedding=query_emb, top_k=3)

print("\nQuery:", query)
print("\nHybrid ranked results:")

for rank, item in enumerate(results, start=1):
    chunk = item["chunk"]
    print(f"{rank}. {chunk['chunk_id']} | final={item['score']:.4f} "
          f"(bm25={item['bm25_score']:.4f}, faiss={item['faiss_score']:.4f})")
    print(f"   text: {chunk['text']}")
