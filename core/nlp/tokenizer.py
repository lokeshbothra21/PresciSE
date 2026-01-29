import re
from typing import List

def tokenize(text: str) -> List[str]:
    """
    Basic scientific-safe tokenizer.
    Preserves symbols, numbers, and abbreviations.
    """
    tokens = re.findall(r"[A-Za-z0-9\-\_\.]+", text)
    return tokens
