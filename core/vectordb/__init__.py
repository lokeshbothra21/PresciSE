#from core.vectordb import FAISSRetriever
from .faiss_index import FAISSIndex
from .faiss_retriever import FAISSRetriever

__all__ = ["FAISSIndex", "FAISSRetriever"]
