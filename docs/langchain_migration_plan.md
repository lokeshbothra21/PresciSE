# LangChain/LangGraph Migration Plan

## My Understanding of Your Requirements

### ✅ What Stays the SAME (Phase 1 - Algorithmic Processing)

**Keep ALL of these custom implementations** (NO changes):

**Document Processing:**
- ✅ PDF loading (`core/ingestion/pdf_loader.py`)
- ✅ Document schema (`core/ingestion/document_schema.py`)
- ✅ Chunking (`core/chunking/pdf_chunker.py`)

**NLP & Retrieval:**
- ✅ Tokenization (`core/nlp/tokenizer.py`)
- ✅ Keyword extraction (`core/nlp/keyword_extraction.py`)
- ✅ Metadata extraction (`core/nlp/metadata_extractor.py`)
- ✅ BM25 retrieval (`core/retrieval/bm25_retriever.py`)
- ✅ FAISS retrieval (`core/vectordb/faiss_retriever.py`)
- ✅ Hybrid retrieval (`core/retrieval/hybrid_retriever.py`)
- ✅ **Dynamic router** (`core/retrieval/search_router.py` - Qwen2.5)
- ✅ SPECTER embeddings (`core/embeddings/embedder.py`)
- ✅ Persistent indexes (`core/persistence/index_manager.py`)

**The entire retrieval flow remains UNCHANGED!**

---

### 🔄 What CHANGES (Phase 2 - Agentic AI)

**Migrate ONLY these components to LangChain/LangGraph:**

#### Current Custom Implementation (to be replaced):

**1. LLM Wrapper** - [`core/llm/gemini_client.py`](file:///c:/Users/ShreyasKrishnaMore/.vscode/Programmes/PresciSE/core/llm/gemini_client.py)
```python
class GeminiClient:
    def generate(self, prompt: str) -> str:
        # Direct Gemini API call
```

**2. Agent** - [`core/agent/scientific_answer_agent.py`](file:///c:/Users/ShreyasKrishnaMore/.vscode/Programmes/PresciSE/core/agent/scientific_answer_agent.py)
```python
class ScientificAnswerAgent:
    def answer(self, query: str, retrieved_chunks: List[Dict]) -> str:
        # Custom prompt building + LLM call
```

**3. Prompts** - [`core/agent/prompts.py`](file:///c:/Users/ShreyasKrishnaMore/.vscode/Programmes/PresciSE/core/agent/prompts.py)
```python
def build_scientific_prompt(query: str, evidence_blocks: str) -> str:
    # Manual prompt template
```

---

## Proposed Architecture After Migration

### Flow Diagram:

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 1 (Unchanged)                      │
│                    Pure Algorithmic Processing                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PDF → Docling → Chunks → Embeddings (SPECTER)                  │
│                      ↓                                           │
│                 BM25 + FAISS Indexes                             │
│                      ↓                                           │
│  Query → Tokenize → Qwen2.5 Router (dynamic weights)            │
│                      ↓                                           │
│            Hybrid Retrieval (top 15 chunks)                      │
│                      ↓                                           │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       │ Selected chunks + query
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                         PHASE 2 (NEW)                            │
│                    LangChain/LangGraph Agent                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  LangChain Gemini LLM (replaces GeminiClient)             │ │
│  └───────────────────────────────────────────────────────────┘ │
│                       ↓                                          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  LangChain PromptTemplate (replaces build_prompt)         │ │
│  │  - Evidence formatting                                     │ │
│  │  - Citation instructions                                   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                       ↓                                          │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  LangGraph Agent Workflow (replaces ScientificAgent)      │ │
│  │  - Evidence formatter node                                 │ │
│  │  - Answer generator node                                   │ │
│  │  - (Optional) Chain-of-thought reasoning                   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                       ↓                                          │
│                  Final Answer                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Mapping

### 1. **LLM Wrapper**: `GeminiClient` → **LangChain ChatGoogleGenerativeAI**

**Current** (`core/llm/gemini_client.py`):
```python
from google import genai

class GeminiClient:
    def __init__(self, model_name="gemini-2.5-flash-lite"):
        self.client = genai.Client(api_key=...)
    
    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(...)
        return response.text
```

**New** (LangChain):
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
    google_api_key=os.getenv("GEMINI_API_KEY")
)
```

---

### 2. **Prompt Building**: `build_scientific_prompt()` → **LangChain PromptTemplate**

**Current** (`core/agent/prompts.py`):
```python
def build_scientific_prompt(query: str, evidence_blocks: str) -> str:
    return f"""
You are PresciSE: a scientific assistant.
USER QUERY: {query}
EVIDENCE: {evidence_blocks}
...
"""
```

**New** (LangChain):
```python
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["query", "evidence"],
    template="""
You are PresciSE: a scientific assistant.

TASK:
Answer the user query using ONLY the evidence provided.

USER QUERY:
{query}

EVIDENCE (retrieved chunks):
{evidence}

INSTRUCTIONS:
- Write a clear, technical explanation.
- Cite sources using <doc_id, page X>
- Do not invent facts.
"""
)
```

---

### 3. **Agent Workflow**: `ScientificAnswerAgent` → **LangGraph StateGraph**

**Current** (`core/agent/scientific_answer_agent.py`):
```python
class ScientificAnswerAgent:
    def answer(self, query: str, retrieved_chunks: List[Dict]) -> str:
        # Format evidence
        evidence_text = self._format_evidence(retrieved_chunks)
        
        # Build prompt
        prompt = build_scientific_prompt(query, evidence_text)
        
        # Call LLM
        return self.llm.generate(prompt)
```

**New** (LangGraph):
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    query: str
    retrieved_chunks: List[Dict]
    formatted_evidence: str
    final_answer: str

def format_evidence_node(state: AgentState) -> AgentState:
    """Node 1: Format retrieved chunks into citations"""
    chunks = state["retrieved_chunks"]
    blocks = []
    for item in chunks:
        chunk = item["chunk"]
        doc_id = chunk.get("doc_id", "unknown")
        pages = chunk.get("metadata", {}).get("pages", [])
        page_num = pages[0] if pages else "unknown"
        citation = f"[{doc_id}, page {page_num}]"
        blocks.append(f"{citation} {chunk['text']}")
    
    state["formatted_evidence"] = "\\n\\n".join(blocks)
    return state

def answer_generation_node(state: AgentState) -> AgentState:
    """Node 2: Generate answer using LLM"""
    prompt = prompt_template.format(
        query=state["query"],
        evidence=state["formatted_evidence"]
    )
    
    response = llm.invoke(prompt)
    state["final_answer"] = response.content
    return state

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("format_evidence", format_evidence_node)
workflow.add_node("generate_answer", answer_generation_node)
workflow.set_entry_point("format_evidence")
workflow.add_edge("format_evidence", "generate_answer")
workflow.add_edge("generate_answer", END)

agent = workflow.compile()
```

---

## File Changes Required

### Files to MODIFY:

#### 1. `core/llm/gemini_client.py` → Replace with LangChain wrapper
**New content**: Use `ChatGoogleGenerativeAI` from `langchain-google-genai`

#### 2. `core/agent/prompts.py` → Replace with LangChain templates
**New content**: Use `PromptTemplate` from `langchain.prompts`

#### 3. `core/agent/scientific_answer_agent.py` → Replace with LangGraph agent
**New content**: Use `StateGraph` from `langgraph.graph`

#### 4. `scripts/pdf_demo.py` → Update agent initialization
**Change**:
```python
# Old
from core.llm import GeminiClient
from core.agent import ScientificAnswerAgent

llm = GeminiClient()
agent = ScientificAnswerAgent(llm)

# New
from core.agent import create_scientific_agent

agent = create_scientific_agent()  # Returns compiled LangGraph
```

#### 5. `requirements.txt` → Already has langchain/langgraph ✅
Just need to add:
```
langchain-google-genai  # For Gemini integration
```

---

## What DOESN'T Change

### ✅ Retrieval Flow (100% Algorithmic - NO Changes):

1. PDF ingestion → Docling
2. Chunking → 900 chars, 150 overlap
3. Metadata extraction → Keywords, NER
4. Embeddings → SPECTER (768-dim)
5. Indexing → BM25 + FAISS (persistent)
6. Query processing → Tokenization
7. **Router** → Qwen2.5 (dynamic weights)
8. Hybrid retrieval → Merge BM25 + FAISS results

**All of this stays EXACTLY as is!**

---

## Interface Compatibility

### Current Interface (in `pdf_demo.py`):

```python
agent.answer(query=query, retrieved_chunks=retrieved)
```

### New Interface (LangGraph):

```python
# Option 1: Keep same interface (wrapper function)
result = agent.invoke({
    "query": query,
    "retrieved_chunks": retrieved
})
final_answer = result["final_answer"]

# Option 2: Create helper function
def answer(query: str, retrieved_chunks: List) -> str:
    result = agent.invoke({
        "query": query,
        "retrieved_chunks": retrieved
    })
    return result["final_answer"]
```

---

## Benefits of LangChain/LangGraph

### Why LangChain for LLM:
- ✅ Standardized LLM interface
- ✅ Built-in retry logic
- ✅ Streaming support
- ✅ Token counting
- ✅ Easier to swap LLMs (Gemini → GPT-4 → Claude)

### Why LangGraph for Agent:
- ✅ **Visual graph structure** (easier to debug)
- ✅ **State management** (track intermediate steps)
- ✅ **Extensibility** (add reasoning, self-critique nodes)
- ✅ **Monitoring** (LangSmith integration)
- ✅ Future: Multi-agent collaboration

---

## Summary

### What I Understood:

**Phase 1 (100% Unchanged)**:
- Everything from PDF → Retrieval → Top 15 chunks
- Custom algorithms ONLY (BM25, FAISS, SPECTER, Router)
- NO LangChain/LangGraph involvement

**Phase 2 (Migrate to LangChain/LangGraph)**:
- LLM wrapper: Custom `GeminiClient` → `ChatGoogleGenerativeAI`
- Prompts: Manual f-strings → `PromptTemplate`
- Agent: Custom `ScientificAnswerAgent` → `StateGraph` workflow

**Key constraint**:
- Keep EXACT same flow
- Only change HOW the agent is implemented
- Same input (query + chunks) → Same output (answer)

---

## Questions Before Implementation

1. **Do you want the LangGraph workflow to be simple** (2 nodes: format → answer)  
   **OR** more complex (add reasoning, self-critique, etc.)?

2. **Keep the same function signature** `agent.answer(query, chunks)`  
   **OR** use LangGraph's native `agent.invoke(state_dict)`?

3. **Should I add LangSmith tracing** for debugging/monitoring?

4. **Any specific LangGraph features** you want (checkpointing, human-in-loop, etc.)?

---

**Does this match your vision? Let me know what needs adjustment!** 🎯
