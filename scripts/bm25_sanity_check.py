from core.retrieval.bm25_retriever import BM25Retriever
from core.nlp.tokenizer import tokenize

print("=== PRESCISE BM25 SANITY CHECK ===")

# Fake document chunks
documents = [
    {
        "chunk_id": "doc1",
        "text": "Michaelis Menten kinetics describes enzyme reaction rates.",
        "tokens": tokenize("Michaelis Menten kinetics describes enzyme reaction rates."),
    },
    {
        "chunk_id": "doc2",
        "text": "Allosteric regulation modifies enzyme activity through binding sites.",
        "tokens": tokenize("Allosteric regulation modifies enzyme activity through binding sites."),
    },
    {
        "chunk_id": "doc3",
        "text": "Neural networks are used in deep learning models.",
        "tokens": tokenize("Neural networks are used in deep learning models."),
    },
]

# Build BM25 retriever
retriever = BM25Retriever(documents)

# Query
query = "enzyme kinetics and regulation"
query_tokens = tokenize(query)

results = retriever.retrieve(query_tokens, top_k=3)

print("\nQuery:", query)
print("\nRanked results:")

for rank, (doc, score) in enumerate(results, start=1):
    print(f"{rank}. {doc['chunk_id']} | score={score:.4f}")
    print(f"   text: {doc['text']}")
