"""
LangChain-based Gemini LLM wrapper for PresciSE.

Replaces custom GeminiClient with LangChain's ChatGoogleGenerativeAI.
"""

import os
import time
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI


class GeminiClient:
    """
    LangChain-based Gemini wrapper with rate limiting.
    
    Uses ChatGoogleGenerativeAI from langchain-google-genai for:
    - Standardized LLM interface
    - Better error handling
    - Streaming support
    - Token counting
    - Easy model swapping
    
    Rate Limit: 10 requests per minute to conserve API quota.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.2):
        """
        Initialize Gemini LLM with LangChain and rate limiting.
        
        Args:
            model_name: Gemini model identifier
            temperature: Sampling temperature (0.0-1.0)
        """
        # Check both GEMINI_API_KEY and GOOGLE_API_KEY (fallback)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.is_mock = False
        
        # Rate limiting: 10 requests per minute
        self.rate_limit = 10  # requests per minute
        self.request_times: List[float] = []
        
        if not api_key:
            print("WARNING: GEMINI_API_KEY not found. Using Mock Mode.")
            self.is_mock = True
            self.llm = None
        else:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=api_key,
                    convert_system_message_to_human=True  # Gemini compatibility
                )
                self.model_name = model_name
            except Exception as e:
                print(f"WARNING: Failed to initialize Gemini Client ({e}). Using Mock Mode.")
                self.is_mock = True
                self.llm = None

    def _wait_for_rate_limit(self):
        """
        Enforce rate limiting by waiting if necessary.
        
        Rate: 10 requests per minute (1 request every 6 seconds).
        """
        current_time = time.time()
        
        # Remove timestamps older than 60 seconds
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # If we've made 10 requests in the last minute, wait
        if len(self.request_times) >= self.rate_limit:
            # Calculate how long to wait
            oldest_request = self.request_times[0]
            wait_time = 60 - (current_time - oldest_request) + 0.1  # Small buffer
            
            if wait_time > 0:
                print(f"[RATE LIMIT] Waiting {wait_time:.1f}s to stay within {self.rate_limit} req/min...")
                time.sleep(wait_time)
                
                # Re-clean request times after waiting
                current_time = time.time()
                self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Record this request
        self.request_times.append(current_time)

    def generate(self, prompt: str) -> str:
        """
        Generate response from LLM with rate limiting.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            Generated text response
        """
        if self.is_mock:
            return (
                "MOCK MODE (MOCK RESPONSE):\n"
                "Based on the retrieved evidence, here is a summary:\n"
                "- The evidence discusses Michaelis-Menten kinetics and its enzymatic reaction rates.\n"
                "- It also mentions Allosteric regulation and conformational shifts.\n"
                "(Note: This is a simulated response because the API key was missing or invalid.)"
            )
        
        try:
            # Enforce rate limit before making request
            self._wait_for_rate_limit()
            
            # LangChain invoke returns AIMessage object
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating response: {e}"