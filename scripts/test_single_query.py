"""
Quick test to see debug output for a single query.
"""

import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from core.agent.query_enhancer import enhance_query

print("Testing query enhancement with debug logging...\n")
print("=" * 80)

result = enhance_query("Which transmission is better: automatic or manual?")

print("\n" + "=" * 80)
print("\nRESULTS:")
print(f"Type: {result['type']}")
print(f"Subqueries: {result['subqueries']}")
print(f"Weights: {result['weights']}")
