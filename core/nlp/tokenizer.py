import re
from typing import List

def tokenize(text: str) -> List[str]:
    """
    Basic scientific-safe tokenizer.
    Preserves symbols, numbers, and abbreviations, including Unicode Greek letters.
    """
    tokens = re.findall(r"[\w\-\.]+", text, re.UNICODE)
    return [t for t in tokens if any(c.isalnum() for c in t)]
