from __future__ import annotations

from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    contract_text: str
    job_id: str
    messages: Annotated[List[AnyMessage], add_messages]
    final_report: str
    # Intermediate results from scanner
    scan_results: List[Dict[str, Any]]
    # Final accumulated results for the chunk
    chunk_results: List[Dict[str, Any]]
