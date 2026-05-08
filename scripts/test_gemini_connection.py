"""
Detailed test with full error output.
"""

import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

print("=" * 80)
print("Testing Gemini API Connection")
print("=" * 80 + "\n")

# Test 1: Check if API key is loaded
import os
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"✓ GEMINI_API_KEY loaded: {api_key[:20]}...{api_key[-4:]}")
else:
    print("✗ GEMINI_API_KEY not found!")
    sys.exit(1)

print()

# Test 2: Try creating Gemini client
try:
    from core.llm import GeminiClient
    print("Creating GeminiClient...")
    llm = GeminiClient(temperature=0.3)
    
    if llm.is_mock:
        print("✗ WARNING: Client is in MOCK mode!")
    else:
        print("✓ GeminiClient created successfully")
        print(f"  Model: {llm.model_name}")
except Exception as e:
    print(f"✗ Error creating client: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Try generating a simple response  
try:
    print("Testing simple generation...")
    response = llm.generate("Say 'Hello from Gemini!'")
    print(f"✓ Generation successful!")
    print(f"  Response: {response[:100]}...")
except Exception as e:
    print(f"✗ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Try classification
try:
    from core.agent.enhancement_tools import classify_query_type
    print("Testing classification tool...")
    result = classify_query_type.invoke({"query": "What makes an F1 car special?"})
    print(f"✓ Classification successful!")
    print(f"  Result: {result}")
except Exception as e:
    print(f"✗ Classification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 5: Try subquery generation
try:
    from core.agent.enhancement_tools import generate_type_a_subqueries
    print("Testing Type A subquery generation...")
    subqueries = generate_type_a_subqueries.invoke({"query": "What makes an F1 car special?"})
    print(f"✓ Subquery generation completed!")
    print(f"  Generated {len(subqueries)} subqueries:")
    for i, sq in enumerate(subqueries, 1):
        print(f"    {i}. {sq}")
except Exception as e:
    print(f"✗ Subquery generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("All tests passed!")
print("=" * 80)
