from core.nlp import process_query

print("=== PRESCISE NLP SANITY CHECK ===")

query = "Compare Michaelis-Menten kinetics vs allosteric enzyme regulation"
result = process_query(query)

print("Query processing result:")
print(result)
