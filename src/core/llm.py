import json
import re
import logging
from typing import List, Dict, Any, Optional

from src.utils.config_loader import load_config
from src.core.types import ClauseAnalysis
from src.utils.prompt_manager import load_risk_standards

# 从统一模块导入 Prompt 模板（包含空上下文处理规则）
from src.core.prompts import MERGED_SCAN_PROMPT, SELF_REFLECTION_PROMPT

logger = logging.getLogger(__name__)

# Prompt 模板已全部移至 src/core/prompts.py



class LLMClient:
    def __init__(self, source: str = "local"):
        self.config = load_config()
        self.source = source
        self._init_config()
        self.risk_standards_text = load_risk_standards()

    def _init_config(self):
        """初始化配置 (不再使用 ChatOllama，改用直接 HTTP 调用)"""
        if self.source == "cloud":
            cfg = self.config.get("llm_cloud_config", {})
        else:
            cfg = self.config.get("llm_config", {})
        
        self.base_url = cfg.get("base_url", "http://localhost:11434")
        self.model_name = cfg.get("model_name", "qwen3:4b-instruct")
        self.temperature = cfg.get("temperature", 0.1)

    def _call_ollama(self, prompt: str) -> str:
        """直接 HTTP 调用 Ollama API（绕过 ChatOllama 的兼容性问题）"""
        import requests
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise

    def analyze_clause(self, clause_text: str, reference_info: str) -> Optional[ClauseAnalysis]:
        """
        使用合并后的 Prompt 分析单个条款。
        
        参数:
            clause_text: 条款原文
            reference_info: 规则引擎匹配到的参考信息
            
        返回:
            ClauseAnalysis 对象，如果无风险则返回 None
        """
        prompt = MERGED_SCAN_PROMPT.format(
            clause=clause_text,
            reference_info=reference_info
        )

        try:
            # 使用直接 HTTP 调用代替 ChatOllama
            content = self._call_ollama(prompt).strip()
            
            # 解析 Markdown 输出
            return self._parse_markdown_output(content, clause_text)
            
        except Exception as e:
            # 增强错误日志
            error_msg = str(e)
            logger.error(f"LLM analysis error: {error_msg}")
            logger.error(f"  → Source: {self.source}")
            logger.error(f"  → Base URL: {self.base_url}")
            logger.error(f"  → Model: {self.model_name}")
            
            return None

    def _parse_markdown_output(self, content: str, original_text: str) -> Optional[ClauseAnalysis]:
        """
        将特定格式的 Markdown 解析为 ClauseAnalysis 对象。
        增强版：支持多种格式变体，提高对小模型输出的容错能力。
        
        预期格式:
        ## 风险：[风险简述]
        - **等级**：[高/低]
        - **维度**：[1-8]
        - **分析**：[...]
        - **法条**：[...]
        - **建议**：[...]
        """
        # --- 多模式匹配风险标题 ---
        # 小模型可能输出多种变体格式，依次尝试匹配
        risk_title_patterns = [
            r"##\s*风险[：:]\s*(.*?)(?:\n|$)",       # 标准格式：## 风险：xxx
            r"#\s*风险[：:]\s*(.*?)(?:\n|$)",        # 单#格式
            r"\*\*风险\*\*[：:]\s*(.*?)(?:\n|$)",    # **风险**：xxx
            r"风险[：:]\s*(.*?)(?:\n|$)",            # 无标记：风险：xxx
        ]
        
        risk_reason = None
        for pattern in risk_title_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                risk_reason = match.group(1).strip()
                break
        
        # 如果所有模式都未匹配到，记录日志并返回 None
        if risk_reason is None:
            logger.warning(f"无法解析风险标题，LLM 输出:\n{content[:500]}")
            return None

        try:
            # --- 宽松正则匹配字段 ---
            # 支持中英文冒号、可选的列表符号、可选的加粗标记
            def extract_field(field_name: str, default: str = "") -> str:
                """通用字段提取器，支持多种格式变体"""
                patterns = [
                    rf"-\s*\*\*{field_name}\*\*[：:]\s*(.*?)(?:\n|$)",   # - **字段**：xxx
                    rf"\*\*{field_name}\*\*[：:]\s*(.*?)(?:\n|$)",       # **字段**：xxx
                    rf"-\s*{field_name}[：:]\s*(.*?)(?:\n|$)",           # - 字段：xxx
                    rf"{field_name}[：:]\s*(.*?)(?:\n|$)",               # 字段：xxx
                ]
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
                return default
            
            risk_level = extract_field("等级", "低")
            dimension = extract_field("维度", "0")
            evidence = extract_field("证据", "")
            deep_analysis = extract_field("分析", "")
            law_reference = extract_field("法条", "")
            suggestion = extract_field("建议", "建议人工复核")

            # --- 清理风险等级 ---
            # 支持高/中/低三级风险
            if "高" in risk_level: 
                risk_level = "高"
            elif "中" in risk_level: 
                risk_level = "中"
            elif "低" in risk_level: 
                risk_level = "低"
            else: 
                risk_level = "低"  # 默认回退到低风险
            
            # --- 清理维度 ID ---
            # 提取纯数字，处理 "4 (违约责任)" 这种情况
            dim_match = re.search(r"(\d+)", dimension)
            dimension = dim_match.group(1) if dim_match else "0"
            
            # --- 证据验证（闭环控制核心） ---
            # 验证 LLM 提取的证据是否真的存在于原文中
            evidence_valid = None
            if evidence and evidence != "无":
                # 清理证据中的「」括号和多余空格
                clean_evidence = evidence.replace("「", "").replace("」", "").strip()
                clean_evidence = re.sub(r'\s+', '', clean_evidence)  # 移除所有空白
                clean_original = re.sub(r'\s+', '', original_text)  # 原文也移除空白
                
                # 检查证据是否在原文中（模糊匹配策略）
                if clean_evidence and len(clean_evidence) > 5:
                    # 策略1: 精确匹配
                    if clean_evidence in clean_original:
                        evidence_valid = True
                    else:
                        # 策略2: 分段匹配 - 将证据拆分成片段，检查>=60%的片段在原文中
                        segments = [clean_evidence[i:i+10] for i in range(0, len(clean_evidence), 10)]
                        segments = [s for s in segments if len(s) >= 5]  # 只保留>=5字符的片段
                        if segments:
                            match_count = sum(1 for seg in segments if seg in clean_original)
                            evidence_valid = (match_count / len(segments)) >= 0.6  # 60%以上匹配即认为有效
                        else:
                            evidence_valid = True  # 证据太短，默认通过
                    
                    # 只有完全不匹配才警告
                    if not evidence_valid and risk_level == "高":
                        logger.warning(f"证据验证失败: '{clean_evidence[:50]}...' 未找到足够匹配")
                        suggestion = f"⚠️ 证据验证存疑，建议人工复核。{suggestion or ''}"

            # --- 强制一致性检查 ---
            # 如果分析文本中包含"不构成该风险点"等明确的低风险表述，强制修正为低风险
            no_risk_phrases = [
                "不构成该风险点",
                "未体现参考风险点",
                "未发现明显法律风险",
                "属于正常",
                "符合法律规定",
                "不涉及",
                "无风险",
                "低风险"
            ]
            for phrase in no_risk_phrases:
                if phrase in deep_analysis or phrase in risk_reason:
                    risk_level = "低"
                    break

            # 构建 ClauseAnalysis 对象
            return ClauseAnalysis(
                clause_text=original_text,
                risk_level=risk_level,
                risk_reason=risk_reason,
                deep_analysis=deep_analysis,
                law_reference=law_reference,
                suggestion=suggestion,
                dimension=dimension,
                evidence=evidence if evidence and evidence != "无" else None,
                evidence_valid=evidence_valid
            )

        except Exception as e:
            logger.error(f"Markdown parsing error: {e}\nContent: {content}")
            return None

    def self_reflect(self, clause_analysis: ClauseAnalysis, reference_info: str = "") -> tuple[str, str]:
        """
        自反思模式：对 LLM 的分析结果进行二次审查。
        
        Args:
            clause_analysis: 第一轮分析结果
            reference_info: 原始规则库参考信息（包含风险内容+后果分析+涉及法律）
            
        Returns:
            (审查结论, 理由) - 结论可能是 "维持"、"降级" 或 "存疑"
        """
        prompt = SELF_REFLECTION_PROMPT.format(
            clause_text=clause_analysis.clause_text,
            reference_info=reference_info or "无匹配规则",
            risk_level=clause_analysis.risk_level,
            risk_reason=clause_analysis.risk_reason,
            evidence=clause_analysis.evidence or "无",
            analysis=clause_analysis.deep_analysis or ""
        )
        
        try:
            # 使用直接 HTTP 调用
            content = self._call_ollama(prompt).strip()
            
            # 增强解析：支持多种格式变体
            # 尝试匹配 "审查结论"、"**审查结论**"、"结论" 等
            # 支持：维持/调级/存疑
            conclusion_patterns = [
                r"[*\s]*审查结论[*\s]*[：:]\s*[【\[]?(维持|调级|降级|存疑)[】\]]?",
                r"[*\s]*结论[*\s]*[：:]\s*[【\[]?(维持|调级|降级|存疑)[】\]]?",
                r"(维持|调级|降级|存疑)",  # 最后兜底直接匹配
            ]
            
            conclusion = None
            for pattern in conclusion_patterns:
                match = re.search(pattern, content)
                if match:
                    conclusion = match.group(1)
                    break
            
            if not conclusion:
                conclusion = "维持"  # 默认维持，避免误判
                logger.warning(f"Self-reflect: 无法解析结论，默认维持。原文: {content[:100]}")
            
            # 尝试匹配理由
            reason_patterns = [
                r"[*\s]*理由[*\s]*[：:]\s*(.*?)(?:\n|$)",
                r"[*\s]*原因[*\s]*[：:]\s*(.*?)(?:\n|$)",
                r"：[【\[]?(维持|降级|存疑)[】\]]?\s*[,，。]?\s*(.*?)(?:\n|$)",
            ]
            
            reason = None
            for pattern in reason_patterns:
                match = re.search(pattern, content)
                if match:
                    reason = match.group(1).strip() if match.lastindex >= 1 else ""
                    if match.lastindex >= 2:
                        reason = match.group(2).strip()
                    if reason:
                        break
            
            if not reason:
                reason = "审查通过" if conclusion == "维持" else "需要人工复核"
            
            return conclusion, reason
            
        except Exception as e:
            logger.error(f"Self-reflection error: {e}")
            return "维持", "自反思调用失败，默认维持"  # 错误时默认维持，不误判

    def unload_model(self):
        """卸载模型 (Ollama 专用)，释放显存。"""
        try:
            import requests
            
            # 发送 keep_alive=0 请求来卸载模型
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "",
                    "keep_alive": 0  # 立即卸载
                },
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"✅ 模型 {self.model_name} 已卸载，显存已释放")
            else:
                logger.warning(f"⚠️ 模型卸载请求返回: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"⚠️ 模型卸载失败: {e}")

