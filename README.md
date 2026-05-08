# PresciSE - Precise Scientific Search Engine

A Retrieval-Augmented Generation (RAG) system for scientific document question-answering using hybrid search (BM25 + FAISS) and LLM-powered synthesis.

## Features

- **PDF Ingestion**: Extract structured content from scientific papers using Docling
- **Intelligent Chunking**: Split documents into semantically meaningful pieces
- **Hybrid Retrieval**: Combines keyword search (BM25) with semantic search (FAISS)
- **AI-Powered Answers**: Uses Gemini to generate grounded, citation-backed responses
- **Scientific Focus**: Optimized for technical and research literature

## Architecture

```
PDF → Docling Parser → Chunks → Embeddings → Hybrid Retrieval → Gemini → Answer
                                   ├─ BM25 (keywords)
                                   └─ FAISS (semantics)
```

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/PresciSE.git
cd PresciSE
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

Get your API key from: https://aistudio.google.com/apikey

## Usage

### Run Full Demo
```bash
python -m scripts.full_demo
```

### Run PDF Demo
Place your PDF files in `data/pdfs/`, then:
```bash
python -m scripts.pdf_demo
```

Or with a custom query:
```bash
python -m scripts.pdf_demo "What is molecular dynamics simulation?"
```

## Project Structure

```
PresciSE/
├── core/                   # Core modules
│   ├── agent/             # LLM agent for answer generation
│   ├── chunking/          # Document chunking strategies
│   ├── embeddings/        # Text embedding models
│   ├── ingestion/         # PDF loading and parsing
│   ├── llm/               # LLM client (Gemini)
│   ├── nlp/               # NLP utilities (tokenization)
│   ├── retrieval/         # Hybrid retrieval system
│   └── vectordb/          # FAISS vector database
├── scripts/               # Demo and utility scripts
├── config/                # Configuration files
├── data/                  # Data directory
│   └── pdfs/             # Place your PDFs here
├── requirements.txt       # Python dependencies
└── .env                  # API keys (not committed)
```

## How It Works

1. **Ingestion**: PDFs are parsed using Docling to extract text, structure, and metadata
2. **Chunking**: Documents are split into ~1200 character chunks with context preservation
3. **Embedding**: Chunks are vectorized using SentenceTransformers (one-time cost)
4. **Indexing**: BM25 (keyword) and FAISS (semantic) indexes are built
5. **Query Processing**: 
   - User query is tokenized and embedded
   - Hybrid search retrieves top-K most relevant chunks
   - Gemini generates a grounded answer using retrieved evidence

## Configuration

Edit `config/app_config.yaml` and `config/retrieval_config.yaml` to customize:
- Embedding model
- Chunk size
- Retrieval weights (BM25 vs FAISS balance)
- Top-K results

## Requirements

- Python 3.10+
- 4GB+ RAM (for embedding models)
- Gemini API key (free tier available)

## License

[Add your license here]

## Contributing

Contributions welcome! Please open an issue or PR.

## Troubleshooting

### "No module named 'docling'"
```bash
pip install docling
```

### "GEMINI_API_KEY not found"
Make sure you've created `.env` and added your API key.

### Out of Memory
Reduce `max_chars` in chunking or process fewer PDFs at once.

## Acknowledgments

- [Docling](https://github.com/DS4SD/docling) for PDF parsing
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [LangChain](https://langchain.com/) for LLM orchestration
