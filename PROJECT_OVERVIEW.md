# PresciSE - Precision Scientific Evidence Retrieval

## What is PresciSE?

PresciSE is an intelligent document question-answering system designed for scientific research papers. It processes PDF documents and provides accurate, citation-backed answers to complex queries using advanced AI retrieval techniques.

---

## Key Features

### 🚀 **Lightning-Fast Search**
- **Two-Phase Retrieval**: Get initial results in ~2ms, refined results in ~50ms
- **Smart Keyword Matching**: Instantly identifies relevant content using AI-extracted keywords
- **Hybrid Search**: Combines keyword matching (BM25) with semantic understanding (FAISS embeddings)

### 💾 **Persistent Indexing**
- **One-Time Processing**: PDFs are indexed once and reused indefinitely
- **Incremental Updates**: Only processes new or modified documents
- **Automatic Cleanup**: Removes data when documents are deleted
- **~30x Faster** subsequent queries (loads from disk instead of re-processing)

### 🎯 **Intelligent Content Understanding**
- **Automatic Keyword Extraction**: Uses TF-IDF and Named Entity Recognition to identify key concepts
- **Context-Aware Chunking**: Breaks documents into semantically meaningful pieces
- **Metadata-Enriched Search**: Each chunk tagged with extracted keywords for rapid filtering

### 📚 **Citation Tracking**
- **Human-Readable Citations**: Answers include `<document_name, page X>` references
- **Source Verification**: Every claim linked to specific document sections
- **Page-Level Precision**: Citations point to exact pages, not just documents

### 🤖 **AI-Powered Answers**
- **Gemini Integration**: Uses Google's Gemini AI for natural language understanding
- **Evidence-Based Responses**: Generates answers strictly from provided document evidence
- **Scientific Rigor**: Designed specifically for research paper comprehension

### 📂 **Multi-Document Support**
- **Batch Processing**: Handles multiple PDFs simultaneously
- **Cross-Document Search**: Finds relevant information across entire document library
- **Unified Index**: Searches all documents in a single query

### 🔄 **Change Detection**
- **File Hash Tracking**: Uses SHA256 to reliably detect document modifications
- **Smart Re-Indexing**: Only updates changed portions of the index
- **Registry Management**: Maintains comprehensive tracking of all processed documents

---

## How It Works (Simplified)

1. **Upload PDFs** → Drop scientific papers into the `data/pdfs/` folder
2. **Automatic Processing** → System extracts text, identifies key concepts, and builds indexes
3. **Ask Questions** → Query in natural language about document content
4. **Get Cited Answers** → Receive accurate responses with source citations

---

## Use Cases

- **Research Literature Review**: Quickly find specific information across multiple papers
- **Evidence Extraction**: Locate exact claims and their supporting evidence
- **Comparative Analysis**: Understand how different papers address the same topic
- **Rapid Reference**: Get instant answers without manually searching PDFs

---

## What Makes PresciSE Special?

**Progressive Results**: See initial matches in milliseconds, then refined results automatically
**Reliability Over Speed**: Uses cryptographic file hashing for accurate change detection
**Zero Configuration**: Automatically extracts keywords and builds indexes without manual setup
**Citation Integrity**: Every answer includes verifiable source references

---

## Current Capabilities

✅ PDF document ingestion and parsing  
✅ Intelligent text chunking with metadata  
✅ Keyword extraction (TF-IDF + NER)  
✅ Hybrid retrieval (BM25 + FAISS)  
✅ Two-phase progressive search  
✅ Persistent index storage  
✅ Incremental document updates  
✅ AI-generated cited answers  
✅ Multi-document search  

---

**Built for**: Researchers, students, and professionals who need fast, accurate information retrieval from scientific literature.

**Optimized for**: Speed, accuracy, and verifiable citations.
