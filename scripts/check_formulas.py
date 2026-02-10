"""
Debug script to check how formulas appear in current chunks.
"""

import sys
sys.path.insert(0, ".")

import pickle
from pathlib import Path

print("="*80)
print("FORMULA EXTRACTION CHECK")
print("="*80 + "\n")

# Load existing chunks
chunks_path = Path("data/index/chunks.pkl")

if not chunks_path.exists():
    print("❌ No chunks found. Run pdf_demo first to build index.")
    sys.exit(1)

with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks\n")

# Look for chunks containing formula-like patterns
import re

# Common formula patterns
formula_patterns = [
    r'[=±×÷∇∫∑∏√∂∞≈≠≤≥]',  # Math symbols
    r'\b[a-zA-Z]\s*=\s*[a-zA-Z0-9]',  # Simple equations like "E = mc"
    r'\d+/\d+',  # Fractions
    r'\^\d+',  # Exponents
    r'd[A-Za-z]/dt',  # Derivatives
    r'\$.*\$',  # LaTeX inline
]

formula_chunks = []
for chunk in chunks:
    text = chunk.get('text', '')
    for pattern in formula_patterns:
        if re.search(pattern, text):
            formula_chunks.append(chunk)
            break

print(f"Found {len(formula_chunks)} chunks containing formula-like content")
print("-"*80 + "\n")

# Show first 5 formula-containing chunks
for i, chunk in enumerate(formula_chunks[:5]):
    print(f"[Chunk {i+1}] {chunk.get('chunk_id', 'unknown')[:50]}...")
    print(f"Doc: {chunk.get('doc_id', 'unknown')}")
    text = chunk.get('text', '')
    print(f"Text ({len(text)} chars):")
    print(text[:500])
    print("\n" + "-"*80 + "\n")

print("\n" + "="*80)
print("DONE")
print("="*80)
