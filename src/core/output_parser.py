"""
统一的输出解析器模块

解析 LLM 的 Markdown 格式输出，供以下模块统一调用：
- engine.py (实际工作流)
- ablation_benchmark.py (评测脚本)
- llm.py (LLM 客户端)
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedResult:
    """解析后的分析结果（通用版）"""
    risk_level: str = "未知"           # 风险等级：高/中/低
    risk_name: str = ""                # 风险名称/简述
    evidence: str = ""                 # 证据摘录
    analysis: str = ""                 # 详细分析
    law_reference: str = ""            # 法条引用
    suggestion: str = ""               # 修改建议
    parse_success: bool = False        # 解析是否成功
    raw_output: str = ""               # 原始 LLM 输出
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "risk_level": self.risk_level,
            "risk_name": self.risk_name,
            "evidence": self.evidence,
            "analysis": self.analysis,
            "law_reference": self.law_reference,
            "suggestion": self.suggestion,
            "parse_success": self.parse_success,
        }


def parse_markdown_output(content: str) -> ParsedResult:
    """
    解析 LLM 的 Markdown 格式输出
    
    预期格式：
    ## 风险：[风险简述]
    - **等级**：[高/中/低]
    - **证据**：[证据摘录]
    - **分析**：[详细分析]
    - **法条**：[法律依据]
    - **建议**：[修改建议]
    
    Args:
        content: LLM 的原始输出文本
        
    Returns:
        ParsedResult: 解析后的结构化结果
    """
    result = ParsedResult(raw_output=content)
    
    if not content:
        return result
    
    try:
        # 1. 解析风险名称/简述
        risk_name_patterns = [
            r"##\s*风险[：:]\s*(.+?)(?:\n|$)",
            r"风险[：:]\s*(.+?)(?:\n|$)",
        ]
        for pattern in risk_name_patterns:
            match = re.search(pattern, content)
            if match:
                result.risk_name = match.group(1).strip()
                break
        
        # 2. 解析风险等级（支持多种格式变体）
        level_patterns = [
            r"\*\*等级\*\*[：:]\s*(高|中|低)",
            r"等级[：:]\s*\*\*(高|中|低)\*\*",
            r"等级[：:]\s*(高|中|低)",
            r"\*\*(高|中|低)\*\*风险",
            r"(高|中|低)\s*风险",
        ]
        for pattern in level_patterns:
            match = re.search(pattern, content)
            if match:
                result.risk_level = match.group(1)
                break
        
        # 3. 解析证据（支持「」括起和普通格式）
        evidence_patterns = [
            r"\*\*证据\*\*[：:]\s*(.*?)(?=\n\s*-|\n\s*\*\*|$)",
            r"证据[：:]\s*「(.+?)」",
            r"证据[：:]\s*(.*?)(?=\n|$)",
        ]
        for pattern in evidence_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                result.evidence = match.group(1).strip()
                break
        
        # 4. 解析分析
        analysis_patterns = [
            r"\*\*分析\*\*[：:]\s*(.*?)(?=\n\s*-|\n\s*\*\*|$)",
            r"分析[：:]\s*(.*?)(?=\n\s*-|\n\s*\*\*|$)",
        ]
        for pattern in analysis_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                result.analysis = match.group(1).strip()
                break
        
        # 5. 解析法条
        law_patterns = [
            r"\*\*法条\*\*[：:]\s*(.*?)(?=\n\s*-|\n\s*\*\*|$)",
            r"法条[：:]\s*(.*?)(?=\n\s*-|\n\s*\*\*|$)",
        ]
        for pattern in law_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                result.law_reference = match.group(1).strip()
                break
        
        # 6. 解析建议
        suggestion_patterns = [
            r"\*\*建议\*\*[：:]\s*(.*?)(?=\n\s*-|\n\s*\*\*|---|$)",
            r"建议[：:]\s*(.*?)(?=\n\s*-|\n\s*\*\*|---|$)",
        ]
        for pattern in suggestion_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                result.suggestion = match.group(1).strip()
                break
        
        # 7. 判断解析是否成功（必须有有效的风险等级）
        result.parse_success = result.risk_level in ["高", "中", "低"]
        
    except Exception as e:
        logger.warning(f"Parse markdown output error: {e}")
    
    return result


def parse_reflection_output(content: str) -> tuple:
    """
    解析自反思模式的输出
    
    预期格式：
    - **审查结论**：[维持/调级/存疑]
    - **理由**：[一句话说明]
    
    Returns:
        tuple: (结论, 理由)
    """
    conclusion = "维持"
    reason = ""
    
    if not content:
        return conclusion, reason
    
    try:
        # 解析审查结论
        conclusion_patterns = [
            r"\*\*审查结论\*\*[：:]\s*(维持|调级|降级|存疑)",
            r"审查结论[：:]\s*(维持|调级|降级|存疑)",
            r"结论[：:]\s*(维持|调级|降级|存疑)",
        ]
        for pattern in conclusion_patterns:
            match = re.search(pattern, content)
            if match:
                conclusion = match.group(1)
                break
        
        # 解析理由
        reason_patterns = [
            r"\*\*理由\*\*[：:]\s*(.*?)(?:\n|$)",
            r"理由[：:]\s*(.*?)(?:\n|$)",
        ]
        for pattern in reason_patterns:
            match = re.search(pattern, content)
            if match:
                reason = match.group(1).strip()
                break
                
    except Exception as e:
        logger.warning(f"Parse reflection output error: {e}")
    
    return conclusion, reason
