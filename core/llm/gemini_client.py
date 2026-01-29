import os
from google import genai

class GeminiClient:
    """
    Minimal Gemini API wrapper used by PresciSE.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        self.is_mock = False
        
        if not api_key:
            print("WARNING: GEMINI_API_KEY not found. Using Mock Mode.")
            self.is_mock = True
        else:
            try:
                self.client = genai.Client(api_key=api_key)
            except Exception as e:
                print(f"WARNING: Failed to initialize Gemini Client ({e}). Using Mock Mode.")
                self.is_mock = True
        
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        if self.is_mock:
            return (
                "MATCHED_MODE (MOCK RESPONSE):\n"
                "Based on the retrieved evidence, here is a summary:\n"
                "- The evidence discusses Michaelis-Menten kinetics and its enzymatic reaction rates.\n"
                "- It also mentions Allosteric regulation and conformational shifts.\n"
                "(Note: This is a simulated response because the API key was missing or invalid.)"
            )
            
        try:
            response = self.client.models.generate_content(model=self.model_name, contents=prompt)
            return response.text
        except Exception as e:
            return f"Error generating response: {e}"
