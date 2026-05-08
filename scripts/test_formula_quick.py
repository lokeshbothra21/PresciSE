"""Quick import + unit test for formula module."""
import sys
sys.path.insert(0, ".")

from core.formula import FormulaChunk, FormulaExtractor, extract_formulas_from_doc
print("✅ Formula module imports OK")

# Test extractor
e = FormulaExtractor()
results = e.extract("The energy E = mc^2 demonstrates mass-energy equivalence. Also F = ma is fundamental.")
print(f"✅ Extracted {len(results)} formulas from test text:")
for i, (formula, conf) in enumerate(results):
    print(f"  [{i+1}] {formula} (confidence: {conf:.2f})")

# Test normalization
norm = FormulaExtractor.normalize("  E  =  mc**2  ")
print(f'✅ Normalization: "E  =  mc**2" → "{norm}"')

# Test schema
fc = FormulaChunk(
    chunk_id="test_0",
    formula_text="E = mc^2",
    context_text="Einstein mass-energy equivalence",
    doc_id="test"
)
d = fc.to_dict()
print(f"✅ FormulaChunk.to_dict() keys: {list(d.keys())}")
print(f'✅ Display text: {d["text"]}')
print(f'✅ Embedding text: {fc.get_embedding_text()}')

# Test HybridRetriever accepts formula_chunks param
from core.retrieval.hybrid_retriever import HybridRetriever
import inspect
sig = inspect.signature(HybridRetriever.__init__)
params = list(sig.parameters.keys())
assert "formula_chunks" in params, "formula_chunks not in HybridRetriever.__init__"
print(f"✅ HybridRetriever accepts formula_chunks param")

# Test prompts have formula instructions
from core.agent.prompts import SCIENTIFIC_ANSWER_TEMPLATE
assert "FORMULA INSTRUCTIONS" in SCIENTIFIC_ANSWER_TEMPLATE
print("✅ Prompts contain FORMULA INSTRUCTIONS block")

print("\n🎉 ALL CHECKS PASSED")
