import json
import re
import logging
from typing import List, Dict, Any, Optional

from src.utils.config_loader import load_config
from src.core.types import ClauseAnalysis
from src.utils.prompt_manager import load_risk_standards

logger = logging.getLogger(__name__)

# 核心分析 Prompt：从 JSON 改为 Markdown，大幅缩短 Context
MERGED_SCAN_PROMPT = """
你是资深合同律师。请基于【参考信息】和【风险定级标准】对以下合同条款进行合规审查。

### 📄 待审条款
{clause_text}

### 📚 参考信息 (已匹配风险库)
{reference_info}

### 风险定级标准
1. **高风险**：条款明显违反法律强制性规定（导致无效/违法）；或触发后需支付双倍赔偿/行政罚款；或导致核心权益（如无固定期限合同签署权）被剥夺。
2. **中风险**：条款表述模糊、存在解释歧义；或举证责任倒置/过高；或显失公平但未直接违法（属于隐患）。
3. **低风险**：条款存在轻微瑕疵，但法律有兜底规定（自动补位）；或仅为程序性笔误，不造成实质损失。

🟢 正确分析范例（学习此逻辑）
条款：“乙方需服从甲方的加班安排。” 分析：虽然条款未提及加班费，但根据法律规定，加班需经过审批。此条款表述虽强势，但未明确写“不支付加班费”，因此属于中风险（隐患），而非高风险（违法）。

🔴 错误分析范例（禁止此类推断）
条款：“合同期满即终止。” 错误分析：属于违法终止，因为可能是第二次续签。 错误原因：过度推断。在合同未提及续签背景时，不得假设其为第二次合同。

### 📝 写作指令
1. **【相关性强制判断规则（必须严格执行）】**
   进行分析前必须先判断条款是否包含参考信息中风险点的“关键动作或典型措辞”。
   若条款未出现参考风险点的以下内容，则视为“不相关”，必须判定为低风险：
   - **核心动词**：如“修改、变更、调整、解释、免责、解除、终止、违约金、罚则、限制、禁止”等；
   - **主体结构**：如“单方/甲方有权…/保留权利…/未经对方同意”等；
   - **风险词组**：直接来自参考信息的关键词字段。
   
   **若上述任一核心动作均未出现，则**：
   - 直接输出“低风险”；
   - “分析”中写明“条款内容未体现参考风险点的关键行为，因此不构成该风险点”；
   - 不得扩写参考信息，不得引用该风险点的风险逻辑。

2. **若参考信息相关性成立**：则直接扩写参考信息的专家逻辑，形成一段通顺分析，并保留参考信息中的“法律依据”。高风险：必须明确指出违反了哪条法律强制性规定，或指明具体的经济赔偿后果。中风险：重点分析条款的模糊性或举证难度，说明为何会增加后期的沟通/维权成本。低风险：说明法律有兜底或无实质损害。分析必须保持客观、中性，不得使用情绪化措辞。仅根据 analysis_logic 描述其违反法律或损害公平原则，语气需保持专业稳健。
3. **若无参考信息或判定未命中**：基于公平原则简要分析。若无明显风险，直接输出“低风险”。
4. **格式要求**：严禁输出 JSON，仅输出以下 Markdown 格式（不要包含 ```markdown 标记）：

## 风险：[风险简述]
- **等级**：[高/中/低] 
- **证据**：[从条款原文中逐字摘录能证明该风险的关键语句，用「」括起来，如无则填"无"]
- **分析**：[基于参考信息扩写的详细分析，100字以内]
- **法条**：[直接复述参考信息中的法条，若无则留空]
- **建议**：[针对性的修改建议]
---
"""

# 自反思 Prompt（可选模式）- 现在包含原始规则信息
SELF_REFLECTION_PROMPT = """
你是一个法律审查专家。请审查以下AI分析是否正确可靠。

【条款原文】
{clause_text}

【专家规则库参考】
{reference_info}

【风险定级标准】
1. 高风险：违法/无效/双倍赔偿/核心权益剥夺。对抗法律：条款内容显式地与法律强制性规定相抵触（如约定低于最低工资、约定放弃社保）。后果严重：将直接导致行政处罚、双倍赔偿或核心权利灭失。
2. 中风险：模糊/歧义/举证责任倒置/隐形损失。沉默/模糊：条款未对某些事项进行约定（如未约定管辖地、未约定具体发薪日），导致需依据法定规则进行解释或补位。程序瑕疵：条款虽未直接对抗法律，但增加了沟通成本或解释的不确定性，且不会直接导致罚款或赔偿。
3. 低风险：法律自动补位/程序性瑕疵。法律自动补位：条款虽未直接对抗法律，但法律规定了兜底条款（如未约定管辖地，法律规定了劳动争议仲裁委员会为管辖地）。

【AI分析结果】
- 风险等级：{risk_level}
- 风险简述：{risk_reason}
- 证据摘录：{evidence}
- 详细分析：{analysis}

### 审查要点（按优先级）
1. **核心判断**：AI判定的风险点是否在条款原文中有实质体现？（核心动作/关键词需一致）
2. **逻辑合理性**：分析逻辑是否与专家规则库的描述一致？
3. **风险等级**：是否存在明显的过度判定（如将普通条款判为高风险）？（严查：将仅具有模糊性的“中风险”夸大为违法的“高风险”；或将违法的“高风险”误判为“中/低风险”）

### 判定规则
- **维持**：如果证据在条款原文中有语义对应（用词可以不完全相同），且分析合理
- **调级**：证据真实存在，但风险等级判定错误（如将“中风险”误判为“高风险”，或反之）。
- **存疑**：仅当分析明显自相矛盾或完全无法证实时使用

### 输出要求（严格遵守）
- **审查结论**：[维持/调级/存疑]
- **理由**：[一句话说明]
"""


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
            clause_text=clause_text,
            reference_info=reference_info,
            risk_standards=self.risk_standards_text
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

