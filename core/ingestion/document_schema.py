from typing import List, Dict, Optional
from pydantic import BaseModel


class Section(BaseModel):
    section_type: str
    content: str
    page_numbers: List[int]


class Figure(BaseModel):
    caption: str
    page_number: int


class Document(BaseModel):
    document_id: str
    title: Optional[str] = None
    sections: List[Section]
    figures: List[Figure]
    metadata: Dict
