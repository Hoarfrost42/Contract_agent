from typing import List

from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    """Request model for contract analysis."""

    text: str
    llm_source: str = "local"
    deep_reflection: bool = False  # 可选：启用深度反思模式（二次审查）


class ClauseAnalysis(BaseModel):
    """Result structure for each clause."""

    clause: str
    risk_assessment: str
    retrieved_info: str


class AnalyzeResponse(BaseModel):
    """Response payload containing all clause analyses."""

    results: List[ClauseAnalysis]
    report: str
