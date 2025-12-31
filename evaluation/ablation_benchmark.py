"""
消融实验评测脚本 (Ablation Study Benchmark)

支持4种评测模式对比：
- Mode 1: 纯LLM（无Prompt模板，直接输入条款）
- Mode 2: 基础Prompt（有格式化Prompt，无规则引擎）
- Mode 3: 当前工作流（Prompt + 规则引擎）
- Mode 4: 优化工作流（改进Prompt + 规则引擎）

评测指标：
- 风险等级准确率 (Accuracy)
- F1 分数 (Precision/Recall 平衡)
- Jaccard 相似度
- 幻觉率 (引用验证失败比例)
- 解析成功率
"""

import argparse
import asyncio
import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import requests
from src.utils.config_loader import load_config


# ============================================================================
# Ollama 客户端 (使用 requests 替代 httpx，解决 502 兼容性问题)
# ============================================================================

class OllamaClient:
    """直接使用 requests 调用 Ollama API，避免 httpx 兼容性问题"""
    
    def __init__(self, base_url: str, model: str, temperature: float = 0):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
    
    def chat(self, prompt: str, timeout: int = 120) -> str:
        """发送聊天请求"""
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": self.temperature}
                },
                timeout=timeout
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")
    
    async def achat(self, prompt: str, timeout: int = 120) -> str:
        """异步版本（实际使用同步调用，因为 asyncio 环境下可以用 run_in_executor）"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(prompt, timeout))


# ============================================================================
# 评测模式定义
# ============================================================================

class EvalMode:
    """评测模式枚举"""
    RAW_LLM = 1           # 纯LLM，无Prompt模板
    BASIC_PROMPT = 2      # 基础Prompt，无规则引擎
    CURRENT_WORKFLOW = 3  # 当前工作流（Prompt + 规则引擎）
    OPTIMIZED_WORKFLOW = 4  # 优化工作流（改进Prompt + 规则引擎）
    
    @staticmethod
    def name(mode: int) -> str:
        names = {
            1: "纯LLM (Raw)",
            2: "基础Prompt",
            3: "当前工作流",
            4: "优化工作流"
        }
        return names.get(mode, "未知")


# ============================================================================
# Prompt 模板定义
# ============================================================================

# 模式1：纯LLM，最简单的输入
RAW_LLM_PROMPT = """请分析以下合同条款是否存在风险：

{clause}

请按以下格式输出：
## 风险：[风险简述]
- **等级**：[高/低]
- **证据**：[证据摘录]
- **分析**：[分析内容]
- **法条**：[相关法条]
- **建议**：[修改建议]
"""

# 模式2：基础Prompt，有格式约束和角色定义，但无规则引擎
BASIC_PROMPT = """你是一位专业的合同法律顾问，请分析以下合同条款是否存在风险。

### 条款原文
{clause}

### 输出要求
请按以下Markdown格式输出（不要包含```markdown标记）：

## 风险：[风险简述]
- **等级**：[高/低] (仅限二选一)
- **证据**：[从条款原文中逐字摘录能证明该风险的关键语句，用「」括起来]
- **分析**：[详细分析，100字以内]
- **法条**：[相关法条，若无则留空]
- **建议**：[针对性的修改建议]
"""

# 模式3：当前工作流Prompt（从llm.py复制）
CURRENT_WORKFLOW_PROMPT = """你是一位专业的合同法律顾问。请基于以下信息分析条款风险。

### 📌 条款原文
{clause}

### 📚 参考信息 (已匹配风险库)
{reference_info}

### 📝 写作指令
1. **【相关性强制判断规则】**
   进行分析前必须先判断条款是否包含参考信息中风险点的"关键动作或典型措辞"。
   若条款未出现核心动词（如"修改、变更、解除、终止"等）或主体结构（如"单方/甲方有权"等），
   则视为"不相关"，必须判定为低风险。

2. **若参考信息相关性成立**：扩写参考信息的专家逻辑，保留法律依据。
3. **若无参考信息或未命中**：基于公平原则简要分析。

请按以下Markdown格式输出：

## 风险：[风险简述]
- **等级**：[高/低] (仅限二选一)
- **证据**：[从条款原文中逐字摘录，用「」括起来]
- **分析**：[详细分析，100字以内]
- **法条**：[法律依据]
- **建议**：[修改建议]
"""

# 模式4：优化工作流Prompt（加入CoT分步推理）
OPTIMIZED_WORKFLOW_PROMPT = """你是一位专业的合同法律顾问。请按照以下步骤分析条款风险。

### 📌 条款原文
{clause}

### 📚 参考信息 (已匹配风险库)
{reference_info}

### 🔍 分析步骤（Chain-of-Thought）

**第一步：关键要素识别**
- 识别主体：甲方/乙方的权利义务
- 识别动作：权利/义务/限制/禁止
- 识别数字：金额/期限/比例

**第二步：与参考信息对照**
- 检查条款是否包含参考信息中的核心动词或风险措辞
- 若不包含，直接判定为低风险
- 若包含，进入深度分析

**第三步：风险评估**
- 评估权利是否对等
- 评估是否违反法律强制性规定
- 综合给出风险等级

请按以下Markdown格式输出：

## 风险：[风险简述]
- **等级**：[高/低] (仅限二选一)
- **证据**：[从条款原文中逐字摘录，用「」括起来]
- **分析**：[详细分析，100字以内]
- **法条**：[法律依据]
- **建议**：[修改建议]
"""


# ============================================================================
# 解析器
# ============================================================================

@dataclass
class ParsedResult:
    """解析后的分析结果"""
    risk_level: str = "未知"
    evidence: str = ""
    analysis: str = ""
    law_reference: str = ""
    suggestion: str = ""
    parse_success: bool = False
    raw_output: str = ""


def parse_markdown_output(content: str) -> ParsedResult:
    """解析LLM的Markdown输出"""
    result = ParsedResult(raw_output=content)
    
    if not content:
        return result
    
    try:
        # 解析风险等级
        level_patterns = [
            r"\*\*等级\*\*[：:]\s*(高|低)",
            r"等级[：:]\s*(高|低)",
            r"\*\*(高|低)\*\*"
        ]
        for pattern in level_patterns:
            match = re.search(pattern, content)
            if match:
                result.risk_level = match.group(1)
                break
        
        # 解析证据
        evidence_patterns = [
            r"\*\*证据\*\*[：:]\s*(.*?)(?=\n\s*-|\n\s*\*\*|$)",
            r"证据[：:]\s*「(.+?)」"
        ]
        for pattern in evidence_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                result.evidence = match.group(1).strip()
                break
        
        # 解析分析
        analysis_patterns = [
            r"\*\*分析\*\*[：:]\s*(.*?)(?=\n\s*-|\n\s*\*\*|$)",
        ]
        for pattern in analysis_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                result.analysis = match.group(1).strip()
                break
        
        # 解析法条
        law_patterns = [
            r"\*\*法条\*\*[：:]\s*(.*?)(?=\n\s*-|\n\s*\*\*|$)",
        ]
        for pattern in law_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                result.law_reference = match.group(1).strip()
                break
        
        # 解析建议
        suggestion_patterns = [
            r"\*\*建议\*\*[：:]\s*(.*?)(?=\n\s*-|\n\s*\*\*|---|$)",
        ]
        for pattern in suggestion_patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                result.suggestion = match.group(1).strip()
                break
        
        # 判断解析是否成功（至少有风险等级）
        result.parse_success = result.risk_level in ["高", "低"]
        
    except Exception as e:
        print(f"Parse error: {e}")
    
    return result


# ============================================================================
# 评估器
# ============================================================================

@dataclass
class EvalMetrics:
    """评估指标"""
    # 基础指标
    total: int = 0
    correct_risk: int = 0
    correct_reason: int = 0
    parse_success: int = 0
    
    # 混淆矩阵（用于 Precision/Recall/F1）
    true_positive: int = 0   # 预测高风险，实际高风险
    false_positive: int = 0  # 预测高风险，实际低风险
    false_negative: int = 0  # 预测低风险，实际高风险
    true_negative: int = 0   # 预测低风险，实际低风险
    
    # 幻觉检测
    evidence_valid: int = 0
    evidence_invalid: int = 0
    
    # 新增：risk_id 匹配（多标签场景）
    risk_id_match: int = 0
    risk_id_total: int = 0
    
    # 新增：响应时间统计
    total_latency: float = 0.0
    
    def accuracy(self) -> float:
        return self.correct_risk / self.total if self.total > 0 else 0
    
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom > 0 else 0
    
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom > 0 else 0
    
    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    def parse_rate(self) -> float:
        return self.parse_success / self.total if self.total > 0 else 0
    
    def hallucination_rate(self) -> float:
        total_evidence = self.evidence_valid + self.evidence_invalid
        return self.evidence_invalid / total_evidence if total_evidence > 0 else 0
    
    def risk_id_accuracy(self) -> float:
        return self.risk_id_match / self.risk_id_total if self.risk_id_total > 0 else 0
    
    def avg_latency(self) -> float:
        return self.total_latency / self.total if self.total > 0 else 0
    
    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "accuracy": round(self.accuracy(), 4),
            "precision": round(self.precision(), 4),
            "recall": round(self.recall(), 4),
            "f1": round(self.f1(), 4),
            "parse_rate": round(self.parse_rate(), 4),
            "hallucination_rate": round(self.hallucination_rate(), 4),
            "risk_id_accuracy": round(self.risk_id_accuracy(), 4),
            "avg_latency_sec": round(self.avg_latency(), 3),
            "reason_quality": round(self.correct_reason / self.total if self.total > 0 else 0, 4),
        }


def verify_evidence(evidence: str, clause: str) -> bool:
    """验证证据是否存在于原文中"""
    if not evidence or evidence in ["无", "None", ""]:
        return True  # 无证据不算幻觉
    
    # 清理证据文本
    clean_evidence = evidence.replace("「", "").replace("」", "").strip()
    if len(clean_evidence) < 5:
        return True
    
    # 检查证据是否在原文中
    return clean_evidence in clause


# ============================================================================
# 评测执行器
# ============================================================================

class AblationBenchmark:
    """消融实验评测器"""
    
    def __init__(self, mode: int, source: str = "local"):
        self.mode = mode
        self.config = load_config()
        
        # 初始化 LLM (使用 OllamaClient 替代 ChatOllama)
        if source == "local":
            llm_cfg = self.config.get("llm_config", {})
        else:
            llm_cfg = self.config.get("llm_cloud_config", {})
        
        self.llm = OllamaClient(
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
            model=llm_cfg.get("model_name", "qwen3:4b-instruct"),
            temperature=0,
        )
        
        # 模式3和4需要规则引擎
        self.rule_engine = None
        if mode in [EvalMode.CURRENT_WORKFLOW, EvalMode.OPTIMIZED_WORKFLOW]:
            from src.core.rule_engine import RuleEngine
            self.rule_engine = RuleEngine()
        
        # Judge LLM (用于评估论证质量)
        self.judge_llm = self.llm
    
    def get_prompt(self, clause: str, reference_info: str = "") -> str:
        """根据模式获取Prompt"""
        if self.mode == EvalMode.RAW_LLM:
            return RAW_LLM_PROMPT.format(clause=clause)
        elif self.mode == EvalMode.BASIC_PROMPT:
            return BASIC_PROMPT.format(clause=clause)
        elif self.mode == EvalMode.CURRENT_WORKFLOW:
            return CURRENT_WORKFLOW_PROMPT.format(clause=clause, reference_info=reference_info or "无")
        elif self.mode == EvalMode.OPTIMIZED_WORKFLOW:
            return OPTIMIZED_WORKFLOW_PROMPT.format(clause=clause, reference_info=reference_info or "无")
        else:
            return RAW_LLM_PROMPT.format(clause=clause)
    
    def get_reference_info(self, clause: str) -> str:
        """获取规则引擎的参考信息"""
        if self.rule_engine is None:
            return ""
        
        try:
            matched_rule, confidence, match_source = self.rule_engine.match_risk(clause)
            if matched_rule:
                return f"**{matched_rule.get('risk_name', '')}**\n{matched_rule.get('analysis_logic', '')}\n"
            elif match_source.startswith("keyword_fallback:"):
                keyword = match_source.split(":", 1)[1]
                return f"【关键词预警】检测到高危关键词：\"{keyword}\"，建议谨慎判断。\n"
        except Exception as e:
            print(f"Rule engine error: {e}")
        
        return ""
    
    async def analyze_clause(self, clause: str) -> ParsedResult:
        """分析单个条款"""
        # 获取参考信息（模式3和4）
        reference_info = self.get_reference_info(clause)
        
        # 构建Prompt
        prompt = self.get_prompt(clause, reference_info)
        
        try:
            # 使用 OllamaClient 的 achat 方法
            content = await self.llm.achat(prompt)
            
            # 解析输出
            result = parse_markdown_output(content)
            return result
            
        except Exception as e:
            print(f"LLM error: {e}")
            return ParsedResult()
    
    async def evaluate_reason(self, clause: str, gt_keywords: List[str], ai_reason: str) -> int:
        """使用LLM-as-a-Judge评估论证质量"""
        if not ai_reason or not gt_keywords:
            return 0
        
        judge_prompt = f"""你是一个公正的法律评估专家。请评估 AI 生成的风险分析理由是否准确。

### 评估输入
- **条款原文**: {clause}
- **标准答案关键词**: {", ".join(gt_keywords)}
- **AI 生成理由**: {ai_reason}

### 评分标准
- **1分 (准确)**: AI 的理由包含了标准答案中的核心关键词或逻辑。
- **0分 (错误)**: AI 的理由完全偏离，或未识别出核心风险。

### 输出格式
仅输出一个数字：1 或 0
"""
        
        try:
            content = await self.judge_llm.achat(judge_prompt)
            return 1 if "1" in content.strip() else 0
        except Exception as e:
            print(f"Judge error: {e}")
            return 0
    
    async def evaluate_single(self, item: Dict[str, Any], metrics: EvalMetrics) -> Dict[str, Any]:
        """评估单个样本"""
        import time
        
        text = item.get("text", "")
        gt = item.get("ground_truth", {})
        original_data = item.get("original_data", {})  # LLM 数据集的原始数据
        
        # 记录开始时间
        start_time = time.time()
        
        # 分析条款
        result = await self.analyze_clause(text)
        
        # 记录响应时间
        latency = time.time() - start_time
        metrics.total_latency += latency
        
        metrics.total += 1
        
        # 解析成功率
        if result.parse_success:
            metrics.parse_success += 1
        
        # 风险等级评估
        gt_risk = gt.get("risk_level", "")
        if gt_risk == "中":
            gt_risk = "低"  # 当前系统不支持中风险
        
        pred_risk = result.risk_level
        is_risk_correct = (gt_risk == pred_risk) or (gt_risk in pred_risk)
        
        if is_risk_correct:
            metrics.correct_risk += 1
        
        # 混淆矩阵更新
        if pred_risk == "高" and gt_risk == "高":
            metrics.true_positive += 1
        elif pred_risk == "高" and gt_risk != "高":
            metrics.false_positive += 1
        elif pred_risk != "高" and gt_risk == "高":
            metrics.false_negative += 1
        else:
            metrics.true_negative += 1
        
        # risk_id 匹配评估（针对 LLM 生成的数据集）
        expected_risks = original_data.get("expected_risks", [])
        if expected_risks:
            # 有预期风险，检查是否正确识别
            metrics.risk_id_total += 1
            # 如果预测为高风险且样本确实包含风险，算匹配成功
            if pred_risk == "高" and gt_risk == "高":
                metrics.risk_id_match += 1
        
        # 证据验证（幻觉检测）
        if verify_evidence(result.evidence, text):
            metrics.evidence_valid += 1
        else:
            metrics.evidence_invalid += 1
        
        # 论证质量评估（如果有 reason_keywords）
        reason_keywords = gt.get("reason_keywords", [])
        if reason_keywords:
            reason_score = await self.evaluate_reason(text, reason_keywords, result.analysis)
            if reason_score:
                metrics.correct_reason += 1
        
        return {
            "id": item.get("id"),
            "prediction": {
                "risk_level": result.risk_level,
                "evidence": result.evidence,
                "analysis": result.analysis,
                "parse_success": result.parse_success,
                "latency": round(latency, 3),
            },
            "ground_truth": gt,
            "correct_risk": is_risk_correct,
            "evidence_valid": verify_evidence(result.evidence, text),
        }


def convert_llm_dataset_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 LLM 生成的数据集格式转换为 benchmark 期望的格式
    
    LLM 格式:
    {
        "id": "GENERAL_001_pos_1",
        "contract_text": "...",
        "expected_risks": [{"risk_id": "...", "risk_name": "...", ...}],
        "case_type": "positive/negative/boundary"
    }
    
    Benchmark 格式:
    {
        "id": "...",
        "text": "...",
        "ground_truth": {"risk_level": "高/低", "reason_keywords": [...]},
        "original_data": {...}  # 保留原始数据用于 risk_id 匹配
    }
    """
    # 检测是否为新格式
    if "contract_text" in item:
        expected_risks = item.get("expected_risks", [])
        case_type = item.get("case_type", "")
        
        # 确定风险等级：positive 且有 expected_risks 为高风险
        if case_type == "positive" and expected_risks:
            risk_level = "高"
        else:
            risk_level = "低"
        
        # 提取关键词作为 reason_keywords
        reason_keywords = []
        for risk in expected_risks:
            if risk.get("risk_name"):
                reason_keywords.append(risk["risk_name"])
        
        return {
            "id": item.get("id", ""),
            "text": item.get("contract_text", ""),
            "ground_truth": {
                "risk_level": risk_level,
                "reason_keywords": reason_keywords,
            },
            "original_data": item,  # 保留原始数据用于 risk_id 匹配
        }
    
    # 已经是旧格式，直接返回
    return item


async def run_ablation_benchmark(
    data_path: str = None,
    mode: int = 2,
    limit: int = None,
    source: str = "local",
    log_callback=None,
    dataset: List[Dict[str, Any]] = None  # 支持传入预加载的数据集
) -> Dict[str, Any]:
    """
    运行消融实验评测
    
    Args:
        data_path: 数据集路径 (当 dataset 为 None 时必须提供)
        mode: 评测模式 (1-4)
        limit: 样本数量限制 (仅当 dataset 为 None 时生效)
        source: LLM 来源 (local/cloud)
        log_callback: 日志回调函数
        dataset: 预加载的数据集 (用于消融实验中控制变量)
    """
    
    def log(msg):
        print(msg)
        if log_callback:
            log_callback(msg)
    
    log(f"\n{'='*60}")
    log(f"🧪 消融实验评测 - 模式 {mode}: {EvalMode.name(mode)}")
    log(f"{'='*60}")
    
    # 如果没有传入数据集，则从文件加载
    if dataset is None:
        if not data_path or not os.path.exists(data_path):
            log(f"Error: Data file not found at {data_path}")
            return None
        
        dataset = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_item = json.loads(line)
                    converted_item = convert_llm_dataset_item(raw_item)
                    dataset.append(converted_item)
        
        # 仅在独立运行时进行随机采样
        total_samples = len(dataset)
        if limit and limit < total_samples:
            import random
            dataset = random.sample(dataset, limit)
            log(f"📊 随机采样 {limit} 条（共 {total_samples} 条可用）")
    
    log(f"📊 实际评测样本数: {len(dataset)}")
    
    # 统计数据集信息
    positive_count = sum(1 for d in dataset if d.get("ground_truth", {}).get("risk_level") == "高")
    negative_count = len(dataset) - positive_count
    log(f"📊 高风险样本: {positive_count}, 低风险样本: {negative_count}")
    
    # 初始化评测器
    benchmark = AblationBenchmark(mode=mode, source=source)
    metrics = EvalMetrics()
    results = []
    
    # 逐个评测
    for i, item in enumerate(dataset):
        log(f"评测进度: {i+1}/{len(dataset)} - {item.get('id', 'unknown')}")
        result = await benchmark.evaluate_single(item, metrics)
        results.append(result)
    
    # 输出报告
    log(f"\n{'='*60}")
    log(f"📈 评测结果 - 模式 {mode}: {EvalMode.name(mode)}")
    log(f"{'='*60}")
    
    metrics_dict = metrics.to_dict()
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            # latency 使用秒数格式，其他使用百分比格式
            if "latency" in key:
                log(f"  {key}: {value:.3f}s")
            else:
                log(f"  {key}: {value:.2%}")
        else:
            log(f"  {key}: {value}")
    
    return {
        "mode": mode,
        "mode_name": EvalMode.name(mode),
        "metrics": metrics_dict,
        "results": results,
    }


async def run_full_ablation_study(
    data_path: str,
    modes: List[int] = None,
    limit: int = None,
    source: str = "local"
) -> Dict[str, Any]:
    """运行完整消融实验（所有模式对比，使用相同样本控制变量）"""
    
    if modes is None:
        modes = [1, 2, 3, 4]  # 默认运行所有模式
    
    print("\n" + "="*70)
    print("🔬 消融实验 (Ablation Study) - 多模式对比评测")
    print("="*70)
    
    # ========== 统一加载和采样数据（控制变量） ==========
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return {}
    
    dataset = []
    boundary_count = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_item = json.loads(line)
                # 过滤掉 boundary 类型的测试用例（没有明确的预期结果）
                case_type = raw_item.get("case_type", "")
                if case_type == "boundary":
                    boundary_count += 1
                    continue
                converted_item = convert_llm_dataset_item(raw_item)
                dataset.append(converted_item)
    
    if boundary_count > 0:
        print(f"📊 已过滤 {boundary_count} 条 boundary 用例（无明确预期结果）")
    
    total_samples = len(dataset)
    if limit and limit < total_samples:
        import random
        dataset = random.sample(dataset, limit)
        print(f"📊 统一随机采样 {limit} 条（共 {total_samples} 条可用）")
    
    print(f"📊 所有模式将使用相同的 {len(dataset)} 条样本")
    
    # 打印样本 ID 以便验证
    sample_ids = [item.get("id", "unknown") for item in dataset[:5]]
    print(f"📊 样本 ID 预览: {sample_ids}{'...' if len(dataset) > 5 else ''}")
    
    all_results = {}
    
    # ========== 对每个模式运行评测（传入相同数据集） ==========
    for mode in modes:
        result = await run_ablation_benchmark(
            mode=mode,
            source=source,
            dataset=dataset  # 传入预加载的数据集
        )
        if result:
            all_results[f"mode_{mode}"] = result
    
    # 生成对比报告
    print("\n" + "="*70)
    print("📊 模式对比汇总")
    print("="*70)
    
    # 表头
    print(f"{'指标':<20}", end="")
    for mode in modes:
        print(f"{EvalMode.name(mode):<18}", end="")
    print()
    print("-" * (20 + 18 * len(modes)))
    
    # 指标对比 (移除维度相关指标)
    metric_keys = ["accuracy", "f1", "precision", "recall", "parse_rate", "hallucination_rate", "risk_id_accuracy", "avg_latency_sec"]
    for key in metric_keys:
        print(f"{key:<20}", end="")
        for mode in modes:
            mode_key = f"mode_{mode}"
            if mode_key in all_results:
                value = all_results[mode_key]["metrics"].get(key, 0)
                # latency 使用秒数格式
                if "latency" in key:
                    print(f"{value:.3f}s".ljust(18), end="")
                else:
                    print(f"{value:.2%}".ljust(18), end="")
            else:
                print(f"{'N/A':<18}", end="")
        print()
    
    # 保存结果 (使用脚本目录的绝对路径)
    script_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = script_dir / f"ablation_results_{timestamp}.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存至: {output_path}")
    
    # ========== 生成图表 ==========
    try:
        from evaluation.chart_generator import generate_ablation_charts, generate_combined_chart
        
        print("\n📊 正在生成评测图表...")
        chart_paths = generate_ablation_charts(all_results, timestamp=timestamp)
        combined_path = generate_combined_chart(all_results, timestamp=timestamp)
        
        # 将图表路径添加到结果中
        all_results["chart_paths"] = chart_paths
        all_results["combined_chart"] = combined_path
        all_results["timestamp"] = timestamp
        
        # 更新保存的 JSON（包含图表路径）
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"📊 已生成 {len(chart_paths)} 张图表 + 1 张综合图")
    except ImportError as e:
        print(f"⚠️ 图表生成失败（请确保安装 matplotlib）: {e}")
    except Exception as e:
        print(f"⚠️ 图表生成异常: {e}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="消融实验评测脚本")
    parser.add_argument("--data", type=str, default="evaluation/llm_benchmark_dataset.jsonl",
                        help="评测数据集路径 (支持新旧两种格式)")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3, 4], default=None,
                        help="评测模式 (1-4)，不指定则运行所有模式")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制评测样本数量")
    parser.add_argument("--source", type=str, choices=["local", "cloud"], default="local",
                        help="LLM来源 (local/cloud)")
    
    args = parser.parse_args()
    
    if args.mode:
        # 运行单个模式
        asyncio.run(run_ablation_benchmark(
            data_path=args.data,
            mode=args.mode,
            limit=args.limit,
            source=args.source
        ))
    else:
        # 运行完整消融实验
        asyncio.run(run_full_ablation_study(
            data_path=args.data,
            limit=args.limit,
            source=args.source
        ))
