"""
LangGraph-based scientific answer agent for PresciSE.

Replaces custom ScientificAnswerAgent with LangGraph StateGraph for:
- Visual workflow structure
- State management
- Better debugging
- Extensibility
- Monitoring support
"""

from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from core.llm import GeminiClient
from core.agent.prompts import get_prompt_template


class AgentState(TypedDict):
    """
    State object passed between LangGraph nodes.
    
    Attributes:
        query: User's question
        retrieved_chunks: List of retrieved evidence chunks with scores
        formatted_evidence: Formatted evidence text with citations
        final_answer: Generated answer from LLM
    """
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    formatted_evidence: str
    final_answer: str


class ScientificAnswerAgent:
    """
    LangGraph-based agent for scientific question answering.
    
    Workflow:
        1. format_evidence: Convert retrieved chunks into cited evidence
        2. generate_answer: Use LLM to generate grounded answer
    
    Maintains backward compatibility with original agent interface.
    """

    def __init__(self, llm: GeminiClient):
        """
        Initialize LangGraph workflow with Gemini LLM.
        
        Args:
            llm: Initialized GeminiClient (LangChain-based)
        """
        self.llm = llm
        self.prompt_template = get_prompt_template()
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        self.agent = self.workflow.compile()

    def _build_workflow(self) -> StateGraph:
        """
        Construct LangGraph StateGraph workflow.
        
        Returns:
            StateGraph with nodes and edges configured
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("format_evidence", self._format_evidence_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        
        # Define flow
        workflow.set_entry_point("format_evidence")
        workflow.add_edge("format_evidence", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow

    def _format_evidence_node(self, state: AgentState) -> AgentState:
        """
        Node 1: Format retrieved chunks into cited evidence blocks.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with formatted_evidence populated
        """
        retrieved_chunks = state["retrieved_chunks"]
        blocks = []
        
        for item in retrieved_chunks:
            chunk = item["chunk"]
            # Extract doc_id and page numbers from chunk
            doc_id = chunk.get("doc_id", "unknown")
            pages = chunk.get("metadata", {}).get("pages", [])
            # Use first page number if available
            page_num = pages[0] if pages and pages[0] > 0 else "unknown"
            citation = f"[{doc_id}, page {page_num}]"
            blocks.append(f"{citation} {chunk['text']}")
        
        state["formatted_evidence"] = "\n\n".join(blocks)
        return state

    def _generate_answer_node(self, state: AgentState) -> AgentState:
        """
        Node 2: Generate answer using LLM and formatted evidence.
        
        Args:
            state: Current agent state with formatted evidence
            
        Returns:
            Updated state with final_answer populated
        """
        # Format prompt using LangChain template
        prompt = self.prompt_template.format(
            query=state["query"],
            evidence=state["formatted_evidence"]
        )
        
        # Generate answer using LLM
        answer = self.llm.generate(prompt)
        state["final_answer"] = answer
        
        return state

    def answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate answer for query using retrieved evidence.
        
        This method provides backward compatibility with the original agent interface.
        Internally uses LangGraph workflow.
        
        Args:
            query: User's question
            retrieved_chunks: List of retrieved evidence chunks
            
        Returns:
            Generated answer string
        """
        # Initialize state
        initial_state = {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "formatted_evidence": "",
            "final_answer": ""
        }
        
        # Execute LangGraph workflow
        result = self.agent.invoke(initial_state)
        
        return result["final_answer"]

    def get_workflow_graph(self):
        """
        Get visual representation of the workflow (for debugging).
        
        Returns:
            Graph object that can be visualized
        """
        return self.agent.get_graph()
