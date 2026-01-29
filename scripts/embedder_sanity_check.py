from core.embeddings.embedder import Embedder

print("=== PRESCISE EMBEDDER SANITY CHECK ===")

embedder = Embedder()
vec = embedder.embed_text("Michaelis-Menten kinetics in enzyme catalysis")

print("Embedding dimension:", len(vec))
print("First 5 values:", vec[:5])
