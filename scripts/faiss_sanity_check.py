from core.embeddings import Embedder
from core.vectordb import FAISSRetriever

print("=== PRESCISE FAISS SANITY CHECK ===")

embedder = Embedder()

chunks = [
    {
        "chunk_id": "c1",
        "text": "Michaelis-Menten kinetics describes enzyme reaction rate behavior.",
        "metadata": {"doc_id": "doc1"},
    },
    {
        "chunk_id": "c2",
        "text": "Allosteric regulation changes enzyme activity through conformational shifts.",
        "metadata": {"doc_id": "doc2"},
    },
    {
        "chunk_id": "c3",
        "text": "Neural networks are used for computer vision and NLP tasks.",
        "metadata": {"doc_id": "doc3"},
    },
]

# Embed chunks
texts = [c["text"] for c in chunks]
embs = embedder.embed_texts(texts)

for c, e in zip(chunks, embs):
    c["embedding"] = e

retriever = FAISSRetriever(chunks)

query = "enzyme kinetics"
query_emb = embedder.embed_text(query)

results = retriever.retrieve(query_emb, top_k=3)

print("\nQuery:", query)
print("\nSemantic results:")

for rank, item in enumerate(results, start=1):
    chunk = item["chunk"]
    score = item["score"]
    print(f"{rank}. {chunk['chunk_id']} | score={score:.4f}")
    print(f"   text: {chunk['text']}")
