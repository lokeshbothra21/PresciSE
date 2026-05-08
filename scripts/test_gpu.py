"""
Test GPU availability and embedder performance.
"""

import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

import torch
from core.embeddings import Embedder
import time

print("="*80)
print("GPU AVAILABILITY CHECK")
print("="*80)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  No GPU detected - will use CPU")

print("\n" + "="*80)
print("EMBEDDER TEST")
print("="*80 + "\n")

# Create embedder (will auto-select GPU if available)
embedder = Embedder()

# Test single embedding
test_text = "Molecular dynamics simulation enables computational drug discovery through atomic-level modeling."

print(f"Embedding test text on {embedder.device}...")
start = time.time()
embedding = embedder.embed_text(test_text)
elapsed = time.time() - start

print(f"✅ Embedding generated in {elapsed:.3f}s")
print(f"   Dimension: {len(embedding)}")
print(f"   First 5 values: {embedding[:5]}")

# Test batch embedding
print("\nTesting batch embedding (10 texts)...")
batch_texts = [f"Test scientific text number {i}" for i in range(10)]

start = time.time()
batch_embeddings = embedder.embed_texts(batch_texts)
elapsed = time.time() - start

print(f"✅ Batch embeddings generated in {elapsed:.3f}s")
print(f"   Texts processed: {len(batch_embeddings)}")
print(f"   Speed: {len(batch_embeddings)/elapsed:.2f} texts/sec")

print("\n" + "="*80)
if embedder.device == 'cuda':
    print("🎮 GPU ACCELERATION IS WORKING!")
else:
    print("💻 Running on CPU (GPU not available)")
print("="*80)
