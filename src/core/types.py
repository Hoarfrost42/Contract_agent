from typing import List, Optional
from pydantic import BaseModel, Field

class ClauseAnalysis(BaseModel):
    clause_text: str = Field(description="合同条款原文")
    risk_level: str = Field(default="无", description="风险等级: 高, 中, 低, 或 无")
    risk_reason: str = Field(default="无详细说明", description="风险简述")
    law_reference: str = Field(default="无", description="具体法律引用 (如 '民法典第500条') 或 '无'")
    dimension: str = Field(default="0", description="风险维度 ID (1-8)")
    dimension_tags: List[str] = Field(default_factory=list, description="映射后的风险维度名称列表")
    
    # 证据回溯字段（闭环控制）
    evidence: Optional[str] = Field(default=None, description="从条款原文中摘录的证据")
    evidence_valid: Optional[bool] = Field(default=None, description="证据是否在原文中验证成立")
    
    # 深度分析后填充的字段
    law_content: Optional[str] = Field(default=None, description="引用法条的具体内容")
    deep_analysis: Optional[str] = Field(default=None, description="基于法条的深度分析")
    full_law_name: Optional[str] = Field(default=None, description="法律全称 (如 中华人民共和国民法典)")
    suggestion: Optional[str] = Field(default=None, description="修改建议")
    risk_id: Optional[str] = Field(default=None, description="匹配的风险规则ID")

class ContractAnalysisResult(BaseModel):
    clauses: List[ClauseAnalysis] = Field(description="分析后的条款列表")

