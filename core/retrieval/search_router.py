"""
Dynamic Search Router using Qwen2.5-1.5B-Instruct.

Analyzes query intent and determines optimal BM25/FAISS weights at runtime.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
from typing import Dict, Optional


class SearchRouter:
    """
    Lightweight LLM-based router for dynamic hybrid search weight determination.
    
    Uses Qwen2.5-1.5B-Instruct to analyze queries and output optimal
    BM25 (lexical) vs FAISS (semantic) weights.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        """
        Initialize router with Qwen2.5 model.
        
        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        
        # Default fallback weights
        self.default_weights = {
            "bm25_weight": 0.6,
            "faiss_weight": 0.4,
            "intent": "fallback",
            "reason": "Using default balanced weights"
        }
        
        if self.device == 'cuda':
            print(f"   Router will use GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   Router will use CPU")
    
    def _load(self):
        """Lazy load model to save memory."""
        if self.model is None:
            print(f"   Loading {self.model_name} for query routing...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None,
                trust_remote_code=True
            )
            
            if self.device == 'cpu':
                self.model = self.model.to(self.device)
            
            print(f"  ✅ Router model loaded on {self.device}")
    
    def route(self, query: str) -> Dict[str, any]:
        """
        Analyze query and determine optimal search weights.
        
        Args:
            query: User query string
            
        Returns:
            {
                "bm25_weight": float (0.0-1.0),
                "faiss_weight": float (0.0-1.0),
                "intent": str (category),
                "reason": str (explanation)
            }
        """
        self._load()
        
        prompt = self._build_prompt(query)
        
        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            if self.device == 'cuda':
                inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,  # Low temperature for consistency
                    do_sample=False,  # Deterministic
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse JSON from response
            weights = self._parse_response(response)
            
            # Validate
            self._validate_weights(weights)
            
            return weights
            
        except Exception as e:
            print(f"  ⚠️ Router failed: {e}")
            print(f"  ↪️ Using fallback weights")
            return {
                **self.default_weights,
                "reason": f"Router error: {str(e)[:100]}"
            }
    
    def _build_prompt(self, query: str) -> str:
        """
        Construct prompt for weight determination.
        
        Args:
            query: User query
            
        Returns:
            Formatted prompt string
        """
        return f"""You are a search query analyzer. Output ONLY valid JSON, nothing else.

Query: "{query}"

Intent Categories:
- exact_match: Specific terms/acronyms/definitions
- semantic_search: Conceptual/exploratory questions  
- definition_lookup: "What is X?" queries
- comparison: "Compare X and Y" queries
- methodology: "How does X work?" queries

Examples of CORRECT output:

Query: "What is protein folding?"
{{"bm25_weight": 0.5, "faiss_weight": 0.5, "intent": "definition_lookup", "reason": "Query asks for definition of specific term"}}

Query: "How do proteins fold in cells?"
{{"bm25_weight": 0.3, "faiss_weight": 0.7, "intent": "methodology", "reason": "Query explores process and mechanism"}}

Query: "Compare classical MD with quantum MD"
{{"bm25_weight": 0.3, "faiss_weight": 0.7, "intent": "comparison", "reason": "Query compares two methodologies"}}

Query: "MD simulation acronym"
{{"bm25_weight": 0.8, "faiss_weight": 0.2, "intent": "exact_match", "reason": "Query seeks specific acronym definition"}}
  
CRITICAL RULES:
1. Output ONLY the JSON object
2. NO explanatory text before or after
3. Weights MUST sum to 1.0
4. Use double quotes for strings
5. Numbers must be between 0.0 and 1.0

Now analyze this query and output ONLY JSON:
{query}

JSON:"""
    
    def _parse_response(self, response: str) -> Dict[str, any]:
        """
        Extract JSON from model response.
        
        Args:
            response: Raw model output
            
        Returns:
            Parsed weight dictionary
        """
        # Find JSON block in response
        # Look for { ... } pattern
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        
        if not json_match:
            raise ValueError("No JSON object found in response")
        
        json_str = json_match.group(0)
        
        # Parse JSON
        weights = json.loads(json_str)
        
        return weights
    
    def _validate_weights(self, weights: Dict[str, any]):
        """
        Validate weight dictionary.
        
        Args:
            weights: Parsed weights
            
        Raises:
            ValueError if validation fails
        """
        # Check required keys
        required = ["bm25_weight", "faiss_weight", "intent", "reason"]
        for key in required:
            if key not in weights:
                raise ValueError(f"Missing required key: {key}")
        
        # Check weight ranges
        bm25 = weights["bm25_weight"]
        faiss = weights["faiss_weight"]
        
        if not (0.0 <= bm25 <= 1.0):
            raise ValueError(f"bm25_weight out of range: {bm25}")
        
        if not (0.0 <= faiss <= 1.0):
            raise ValueError(f"faiss_weight out of range: {faiss}")
        
        # Check sum (allow small floating point error)
        total = bm25 + faiss
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"Weights don't sum to ~1.0: {total}")
        
        # Normalize to exactly 1.0
        weights["bm25_weight"] = bm25 / total
        weights["faiss_weight"] = faiss / total
