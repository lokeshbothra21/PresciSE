from typing import Literal

QueryType = Literal[
    "definition",
    "comparison",
    "methodology",
    "exploratory"
]

def classify_query(text: str) -> QueryType:
    text_lower = text.lower()

    if any(k in text_lower for k in ["compare", "difference", "vs"]):
        return "comparison"

    if any(k in text_lower for k in ["how does", "method", "procedure"]):
        return "methodology"

    if any(k in text_lower for k in ["what is", "define", "explain"]):
        return "definition"

    return "exploratory"
