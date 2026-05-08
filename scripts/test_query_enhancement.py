"""
Test script for query enhancement system.

Tests Type A and Type B query enhancement with subquery generation.
Outputs results to a file for review.
"""

import sys
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from core.agent.query_enhancer import enhance_query
from datetime import datetime


def print_section(title, file=None):
    """Print formatted section header."""
    line = "=" * 80
    output = f"\n{line}\n  {title}\n{line}\n"
    print(output)
    if file:
        file.write(output + "\n")


def test_query_enhancement():
    """Test query enhancement with sample queries."""
    
    # Comprehensive test queries covering various scientific domains
    test_queries = [
        # ===== Type A Examples (Exploratory) =====
        {
            "query": "What makes an F1 car special?",
            "expected_type": "Type A",
            "domain": "Engineering"
        },
        {
            "query": "How does molecular dynamics simulation work?",
            "expected_type": "Type A",
            "domain": "Computational Chemistry"
        },
        {
            "query": "What is the importance of protein folding?",
            "expected_type": "Type A",
            "domain": "Biology"
        },
        {
            "query": "Explain how CRISPR gene editing works",
            "expected_type": "Type A",
            "domain": "Genetics"
        },
        {
            "query": "What is the role of machine learning in drug discovery?",
            "expected_type": "Type A",
            "domain": "AI in Medicine"
        },
        {
            "query": "How do antibodies neutralize viruses?",
            "expected_type": "Type A",
            "domain": "Immunology"
        },
        
        # ===== Type B Examples (Multi-Case) =====
        {
            "query": "What are exceptions to the octet rule?",
            "expected_type": "Type B",
            "domain": "Chemistry"
        },
        {
            "query": "How does pH affect enzyme activity?",
            "expected_type": "Type B",
            "domain": "Biochemistry"
        },
        {
            "query": "Compare classical MD with quantum MD simulations",
            "expected_type": "Type B",
            "domain": "Computational Chemistry"
        },
        {
            "query": "What are different protein folding pathways?",
            "expected_type": "Type B",
            "domain": "Structural Biology"
        },
        {
            "query": "Under what conditions does photosynthesis slow down?",
            "expected_type": "Type B",
            "domain": "Plant Biology"
        },
        {
            "query": "What are the different mechanisms of drug resistance?",
            "expected_type": "Type B",
            "domain": "Pharmacology"
        },
        
        # ===== Edge Cases =====
        {
            "query": "What is apoptosis?",
            "expected_type": "Type A",
            "domain": "Cell Biology"
        },
        {
            "query": "Compare aerobic and anaerobic respiration",
            "expected_type": "Type B",
            "domain": "Metabolism"
        },
        {
            "query": "How does neural network training work?",
            "expected_type": "Type A",
            "domain": "Machine Learning"
        }
    ]
    
    # Open output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results/query_enhancement_test_{timestamp}.md"
    
    import os
    os.makedirs("test_results", exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        # Write header
        f.write("# Query Enhancement System Test Results\n\n")
        f.write(f"**Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Queries Tested**: {len(test_queries)}\n\n")
        f.write("---\n\n")
        
        # Test each query
        results_summary = []
        
        for idx, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            expected = test_case["expected_type"]
            domain = test_case["domain"]
            
            # Print to console
            print_section(f"TEST {idx}/{len(test_queries)}: {query}", f)
            
            # Write to file
            f.write(f"## Test {idx}: {domain}\n\n")
            f.write(f"**Query**: {query}\n\n")
            f.write(f"**Expected Type**: {expected}\n\n")
            
            # Enhance query
            try:
                result = enhance_query(query)
                
                # Classification result
                classified_type = result['type'].replace("type_a", "Type A").replace("type_b", "Type B")
                match = "✅ CORRECT" if classified_type == expected else "❌ MISMATCH"
                
                f.write(f"**Classified As**: {classified_type} {match}\n\n")
                
                # Subqueries section
                f.write(f"### Generated Subqueries ({len(result['subqueries'])} total):\n\n")
                
                for i, subquery in enumerate(result['subqueries'], 1):
                    weight = result['weights'].get(f'subquery_{i-1}', 0.7)
                    f.write(f"{i}. **[Weight: {weight}]** {subquery}\n")
                
                f.write("\n")
                
                # Save summary
                results_summary.append({
                    "test_num": idx,
                    "query": query,
                    "expected": expected,
                    "classified": classified_type,
                    "match": match,
                    "num_subqueries": len(result['subqueries'])
                })
                
                # Console output
                print(f"  Domain: {domain}")
                print(f"  Expected: {expected}")
                print(f"  Classified: {classified_type} {match}")
                print(f"\n  Subqueries Generated ({len(result['subqueries'])}):")
                for i, sq in enumerate(result['subqueries'], 1):
                    print(f"    {i}. {sq}")
                print()
                
            except Exception as e:
                f.write(f"**ERROR**: {str(e)}\n\n")
                print(f"  ERROR: {str(e)}\n")
                results_summary.append({
                    "test_num": idx,
                    "query": query,
                    "expected": expected,
                    "classified": "ERROR",
                    "match": "❌ ERROR",
                    "num_subqueries": 0
                })
            
            f.write("---\n\n")
        
        # Write summary table
        f.write("\n## Summary\n\n")
        f.write("| Test # | Expected | Classified | Match | # Subqueries |\n")
        f.write("|--------|----------|------------|-------|-------------|\n")
        
        correct_count = 0
        for summary in results_summary:
            if "✅" in summary["match"]:
                correct_count += 1
            f.write(f"| {summary['test_num']} | {summary['expected']} | {summary['classified']} | {summary['match']} | {summary['num_subqueries']} |\n")
        
        accuracy = (correct_count / len(test_queries)) * 100
        f.write(f"\n**Accuracy**: {correct_count}/{len(test_queries)} ({accuracy:.1f}%)\n")
        
        print_section("TEST SUMMARY", f)
        print(f"  Total Tests: {len(test_queries)}")
        print(f"  Correct Classifications: {correct_count}")
        print(f"  Accuracy: {accuracy:.1f}%")
        print(f"\n  Results saved to: {output_file}\n")
        
    return output_file


if __name__ == "__main__":
    print("=" * 80)
    print("  PresciSE Query Enhancement System - Comprehensive Test")
    print("=" * 80)
    
    output_file = test_query_enhancement()
    
    print("\n" + "=" * 80)
    print(f"  Test Complete! Results saved to: {output_file}")
    print("=" * 80)
