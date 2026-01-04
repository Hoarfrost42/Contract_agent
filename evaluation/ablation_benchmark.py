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
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.config_loader import load_config

# 导入统一模块
from src.core.ollama_client import OllamaClient
from src.core.output_parser import parse_markdown_output, ParsedResult
from src.core.prompts import (
    RAW_LLM_PROMPT,
    BASIC_PROMPT, 
    CURRENT_WORKFLOW_PROMPT,
    OPTIMIZED_WORKFLOW_PROMPT,
    SELF_REFLECTION_PROMPT,
    get_prompt_by_mode,
)
from src.core.reference_retriever import retrieve_reference



# ============================================================================
# OllamaClient 已移至 src/core/ollama_client.py，通过导入使用
# ============================================================================


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
# Prompt 模板已移至 src/core/prompts.py，通过导入使用
# ============================================================================




# ============================================================================
# ParsedResult 和 parse_markdown_output 已移至 src/core/output_parser.py
# ============================================================================


# ============================================================================
# 工具函数
# ============================================================================

def stratified_sample(dataset: List[dict], limit: int) -> List[dict]:
    """分层采样：确保 High/Medium/Low 比例尽量为 1:1:1"""
    high = [d for d in dataset if d.get("ground_truth", {}).get("risk_level") == "高"]
    medium = [d for d in dataset if d.get("ground_truth", {}).get("risk_level") == "中"]
    low = [d for d in dataset if d.get("ground_truth", {}).get("risk_level") == "低"]
    
    per_class = limit // 3
    
    sampled = []
    import random
    
    # 1. 核心采样：每类抽取 limit/3
    sampled.extend(random.sample(high, min(len(high), per_class)))
    sampled.extend(random.sample(medium, min(len(medium), per_class)))
    sampled.extend(random.sample(low, min(len(low), per_class)))
    
    # 2. 补齐剩余：如果总数不足 limit（因除不尽或某类样本不足）
    current_count = len(sampled)
    if current_count < limit:
        # 创建剩余池：所有未被选中的样本
        # 必须确保 dataset 中的元素也是唯一的或者通过 ID 去重
        sampled_ids = {item.get("id") for item in sampled}
        remaining_pool = [d for d in dataset if d.get("id") not in sampled_ids]
        
        needed = limit - current_count
        if remaining_pool:
            sampled.extend(random.sample(remaining_pool, min(len(remaining_pool), needed)))
    
    random.shuffle(sampled)
    return sampled


def parse_reflection_output(content: str) -> dict:
    """解析自反思输出
    
    期望格式：
    审查结论：[维持 / 调级]
    修正建议：[若调级，请写具体等级流向，如"中风险 -> 低风险"；若维持，填"无"]
    理由：[基于审查基准简述理由]
    
    Returns:
        dict: {
            "conclusion": "维持" / "调级",
            "adjustment": "中风险 -> 低风险" / "无",
            "new_level": "高" / "中" / "低" / None,
            "reason": "..."
        }
    """
    result = {
        "conclusion": "维持",
        "adjustment": "无",
        "new_level": None,
        "reason": ""
    }
    
    # 解析审查结论
    conclusion_match = re.search(r'审查结论[：:]\s*\[?\s*(维持|调级)\s*\]?', content)
    if conclusion_match:
        result["conclusion"] = conclusion_match.group(1)
    
    # 解析修正建议
    adjustment_match = re.search(r'修正建议[：:]\s*\[?\s*(.+?)\s*\]?(?:\n|$)', content)
    if adjustment_match:
        adj = adjustment_match.group(1).strip()
        result["adjustment"] = adj
        
        # 提取新的风险等级
        if "低风险" in adj and "->" in adj:
            result["new_level"] = "低"
        elif "中风险" in adj and "->" in adj:
            result["new_level"] = "中"
        elif "高风险" in adj and "->" in adj:
            result["new_level"] = "高"
    
    # 解析理由
    reason_match = re.search(r'理由[：:]\s*\[?\s*(.+?)\s*\]?(?:\n|$)', content, re.DOTALL)
    if reason_match:
        result["reason"] = reason_match.group(1).strip()[:100]  # 截断到100字
    
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
    
    # 加权评分（精确匹配1分，差一级0.5分，差两级0分）
    total_weighted_score: float = 0.0
    
    # 三分类混淆矩阵 (High=0, Medium=1, Low=2)
    # confusion_matrix[actual][predicted]
    conf_matrix: List[List[int]] = field(default_factory=lambda: [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    
    # 混淆矩阵（旧二分类兼容，用于 Precision/Recall/F1）
    true_positive: int = 0   
    false_positive: int = 0  
    false_negative: int = 0  
    true_negative: int = 0   
    
    # ===== 方法一：证据一致性评估（细分幻觉类型）=====
    clause_evidence_valid: int = 0    # 合同条款证据有效
    clause_evidence_invalid: int = 0  # 合同条款证据幻觉
    law_citation_valid: int = 0       # 法律引用有效
    law_citation_invalid: int = 0     # 法律引用幻觉（不存在的法条）
    
    # 旧字段保持兼容
    evidence_valid: int = 0
    evidence_invalid: int = 0
    
    # ===== 方法二：规则触发一致性 =====
    rule_trigger_count: int = 0   # 实际触发的规则数
    rule_target_count: int = 0    # 应该触发的规则数
    rule_correct_count: int = 0   # 正确触发的规则数
    
    # 新增：risk_id 匹配（多标签场景）
    risk_id_match: int = 0
    risk_id_total: int = 0
    
    # ===== 方法三：任务成功率 =====
    task_success_count: int = 0  # 任务完全成功的样本数
    
    # ===== 自反思机制统计 =====
    reflection_calls: int = 0      # 自反思调用次数
    reflection_adjustments: int = 0  # 反思后调级次数
    reflection_maintain: int = 0   # 反思后维持原判次数
    
    # 响应时间统计
    total_latency: float = 0.0
    
    @staticmethod
    def calculate_weighted_score(gt_risk: str, pred_risk: str) -> float:
        """
        计算非对称加权评分 (Asymmetric Weighted Accuracy)：
        - 精确匹配：1.0分
        - 防御性误判（低→中, 中→高）：0.8分（宁可错杀）
        - 风险降级（高→中）：0.4分（漏报扣分更重）
        - 差两级（高↔低）：0.0分（致命漏判零容忍）
        """
        if gt_risk == pred_risk:
            return 1.0
        
        # 非对称权重矩阵: weight_matrix[gt][pred]
        # gt: 高=0, 中=1, 低=2
        asymmetric_weights = {
            ("高", "中"): 0.4,  # 高风险降为中：危险，扣分重
            ("高", "低"): 0.0,  # 高风险降为低：致命漏判
            ("中", "高"): 0.8,  # 中风险升为高：过度谨慎，可接受
            ("中", "低"): 0.4,  # 中风险降为低：有一定风险
            ("低", "中"): 0.8,  # 低风险升为中：防御性误判
            ("低", "高"): 0.5,  # 低风险升为高：过度报警
        }
        
        return asymmetric_weights.get((gt_risk, pred_risk), 0.0)
            
    def update_confusion_matrix(self, gt_risk: str, pred_risk: str):
        """更新三分类混淆矩阵"""
        level_map = {"高": 0, "中": 1, "低": 2}
        gt_idx = level_map.get(gt_risk, 2)   # 默认为低风险
        pred_idx = level_map.get(pred_risk, 2)
        self.conf_matrix[gt_idx][pred_idx] += 1

    def calculate_kappa(self, use_linear: bool = True) -> float:
        """
        计算加权 Kappa
        
        Args:
            use_linear: True=线性权重(LWK), False=二次方权重(QWK)
        
        Linear Weighted Kappa 公式: w_ij = 1 - |i-j| / (N-1)
        对于有序分类更稳健，不会过度惩罚"离群"错误
        """
        n_classes = 3
        weights = [[0.0] * n_classes for _ in range(n_classes)]
        
        for i in range(n_classes):
            for j in range(n_classes):
                if use_linear:
                    # Linear Weighted Kappa
                    weights[i][j] = abs(i - j) / (n_classes - 1)
                else:
                    # Quadratic Weighted Kappa (原版)
                    weights[i][j] = ((i - j) / (n_classes - 1)) ** 2
                
        # 观察矩阵 O (归一化混淆矩阵)
        total = self.total
        if total == 0: return 0.0
        
        observed = [[self.conf_matrix[i][j] / total for j in range(n_classes)] for i in range(n_classes)]
        
        # 期望矩阵 E (边缘分布外积)
        row_sums = [sum(self.conf_matrix[i]) / total for i in range(n_classes)]
        col_sums = [sum(self.conf_matrix[i][j] for i in range(n_classes)) / total for j in range(n_classes)]
        expected = [[row_sums[i] * col_sums[j] for j in range(n_classes)] for i in range(n_classes)]
        
        # 计算 Kappa = 1 - (sum(W*O) / sum(W*E))
        numerator = sum(weights[i][j] * observed[i][j] for i in range(n_classes) for j in range(n_classes))
        denominator = sum(weights[i][j] * expected[i][j] for i in range(n_classes) for j in range(n_classes))
        
        if denominator == 0: return 1.0  # 完全一致
        return 1.0 - (numerator / denominator)

    def calculate_macro_f1(self) -> dict:
        """计算宏平均 F1"""
        f1_scores = []
        precisions = []
        recalls = []
        
        for k in range(3):  # 0:高, 1:中, 2:低
            # TP = conf[k][k]
            tp = self.conf_matrix[k][k]
            # FP = sum(col[k]) - TP
            fp = sum(self.conf_matrix[i][k] for i in range(3)) - tp
            # FN = sum(row[k]) - TP
            fn = sum(self.conf_matrix[k]) - tp
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            
            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)
            
        return {
            "macro_precision": sum(precisions) / 3,
            "macro_recall": sum(recalls) / 3,
            "macro_f1": sum(f1_scores) / 3,
            "class_f1": {"High": f1_scores[0], "Medium": f1_scores[1], "Low": f1_scores[2]},
            "class_precision": {"High": precisions[0], "Medium": precisions[1], "Low": precisions[2]},
            "class_recall": {"High": recalls[0], "Medium": recalls[1], "Low": recalls[2]}
        }
    
    def calculate_high_risk_f2(self) -> float:
        """
        计算高风险类别的 F2-Score (Recall-Oriented)
        
        F2 = (1 + β²) × (P × R) / (β² × P + R)，其中 β = 2
        F2 分数中 Recall 权重是 Precision 的 2 倍，适合"宁可错杀"的风控场景
        """
        # High = index 0 in confusion matrix
        tp = self.conf_matrix[0][0]
        fp = sum(self.conf_matrix[i][0] for i in range(3)) - tp
        fn = sum(self.conf_matrix[0]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        beta = 2  # F2-Score
        if (precision + recall) == 0:
            return 0.0
        
        f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        return f2
    
    def weighted_accuracy(self) -> float:
        """加权准确率（考虑部分匹配）"""
        return self.total_weighted_score / self.total if self.total > 0 else 0
    
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
    
    # ===== 新增：细分幻觉率 =====
    def clause_hallucination_rate(self) -> float:
        """合同条款证据幻觉率"""
        total = self.clause_evidence_valid + self.clause_evidence_invalid
        return self.clause_evidence_invalid / total if total > 0 else 0
    
    def law_hallucination_rate(self) -> float:
        """法律引用幻觉率"""
        total = self.law_citation_valid + self.law_citation_invalid
        return self.law_citation_invalid / total if total > 0 else 0
    
    # ===== 新增：规则触发一致性 =====
    def rule_recall(self) -> float:
        """规则召回率：正确触发 / 应触发"""
        return self.rule_correct_count / self.rule_target_count if self.rule_target_count > 0 else 0
    
    def rule_precision(self) -> float:
        """规则精确率：正确触发 / 实际触发"""
        return self.rule_correct_count / self.rule_trigger_count if self.rule_trigger_count > 0 else 0
    
    # ===== 新增：任务成功率 =====
    def task_success_rate(self) -> float:
        """任务成功率：完全成功的样本 / 总样本"""
        return self.task_success_count / self.total if self.total > 0 else 0
    
    def risk_id_accuracy(self) -> float:
        return self.risk_id_match / self.risk_id_total if self.risk_id_total > 0 else 0
    
    def avg_latency(self) -> float:
        return self.total_latency / self.total if self.total > 0 else 0
    
    def to_dict(self) -> dict:
        macro = self.calculate_macro_f1()
        return {
            "total": self.total,
            # 基础指标
            "accuracy": round(self.correct_risk / self.total, 4) if self.total > 0 else 0,
            "weighted_accuracy": round(self.total_weighted_score / self.total, 4) if self.total > 0 else 0,
            
            # Kappa 指标 (新增 LWK 和 QWK 对比)
            "kappa_linear": round(self.calculate_kappa(use_linear=True), 4),  # 线性加权 Kappa (推荐)
            "kappa_quadratic": round(self.calculate_kappa(use_linear=False), 4),  # 二次方加权 Kappa (对比)
            "kappa": round(self.calculate_kappa(use_linear=True), 4),  # 默认使用 LWK，保持兼容
            
            # 三分类指标
            "macro_precision": round(macro["macro_precision"], 4),
            "macro_recall": round(macro["macro_recall"], 4),
            "macro_f1": round(macro["macro_f1"], 4),
            
            # 高风险 F2-Score (新增 - Recall 优先)
            "high_risk_f2": round(self.calculate_high_risk_f2(), 4),
            
            # 分类别指标
            "class_f1": {
                "High": round(macro["class_f1"]["High"], 4),
                "Medium": round(macro["class_f1"]["Medium"], 4),
                "Low": round(macro["class_f1"]["Low"], 4)
            },
            "class_precision": {
                "High": round(macro["class_precision"]["High"], 4),
                "Medium": round(macro["class_precision"]["Medium"], 4),
                "Low": round(macro["class_precision"]["Low"], 4)
            },
            "class_recall": {
                "High": round(macro["class_recall"]["High"], 4),
                "Medium": round(macro["class_recall"]["Medium"], 4),
                "Low": round(macro["class_recall"]["Low"], 4)
            },
            
            # 混淆矩阵 (用于绘图)
            "conf_matrix": self.conf_matrix,
            
            # 过程指标
            "parse_rate": round(self.parse_success / self.total, 4) if self.total > 0 else 0,
            
            # 幻觉检测
            "hallucination_rate": round(self.evidence_invalid / (self.evidence_valid + self.evidence_invalid), 4) if (self.evidence_valid + self.evidence_invalid) > 0 else 0,
            "clause_hallucination_rate": round(self.clause_evidence_invalid / (self.clause_evidence_valid + self.clause_evidence_invalid), 4) if (self.clause_evidence_valid + self.clause_evidence_invalid) > 0 else 0,
            "law_hallucination_rate": round(self.law_citation_invalid / (self.law_citation_valid + self.law_citation_invalid), 4) if (self.law_citation_valid + self.law_citation_invalid) > 0 else 0,
            
            # 规则触发
            "rule_recall": round(self.rule_correct_count / self.rule_target_count, 4) if self.rule_target_count > 0 else 0,
            "rule_precision": round(self.rule_correct_count / self.rule_trigger_count, 4) if self.rule_trigger_count > 0 else 0,
            
            # 任务成功率
            "task_success_rate": round(self.task_success_count / self.total, 4) if self.total > 0 else 0,
            
            # Risk ID 准确率
            "risk_id_accuracy": round(self.risk_id_match / self.risk_id_total, 4) if self.risk_id_total > 0 else 0,
            
            # 性能
            "avg_latency_sec": round(self.total_latency / self.total, 3) if self.total > 0 else 0,
            "reason_quality": round(self.correct_reason / self.total, 4) if self.total > 0 else 0,
            
            # 自反思机制统计
            "reflection_calls": self.reflection_calls,
            "reflection_adjustments": self.reflection_adjustments,
            "reflection_maintain": self.reflection_maintain,
            "reflection_adjustment_rate": round(self.reflection_adjustments / self.reflection_calls, 4) if self.reflection_calls > 0 else 0,
        }


def verify_evidence(
    evidence: str, 
    clause: str, 
    embedding_model=None, 
    reranker=None,
    threshold: float = 0.7
) -> tuple:
    """验证证据是否存在于原文中（两阶段检索 + 精排）
    
    流程：
    1. 精确子串匹配（快速路径）
    2. Stage 1: BGE-M3 Dense 召回 Top-K 候选
    3. Stage 2: Reranker 精排，判定相似度是否 >= threshold
    4. 回退：模糊匹配（字符重叠）
    
    Args:
        evidence: LLM 输出的证据字段
        clause: 原始合同条款文本
        embedding_model: Sentence Transformer 模型（用于召回）
        reranker: BGE-Reranker 模型（用于精排）
        threshold: 相似度阈值（默认0.7）
    
    Returns:
        tuple: (is_valid: bool, similarity_score: float, match_type: str)
        - is_valid: 证据是否有效
        - similarity_score: 相似度得分（0-1）
        - match_type: 匹配类型（exact/reranker_match/semantic/fuzzy/hallucination）
    """
    if not evidence or evidence in ["无", "None", "", "留空"]:
        return (True, 1.0, "empty")  # 无证据不算幻觉
    
    # 清理证据文本
    clean_evidence = evidence.replace("「", "").replace("」", "").replace("\"", "").strip()
    if len(clean_evidence) < 5:
        return (True, 1.0, "too_short")
    
    # 1. 精确匹配（子串匹配）- 快速路径
    if clean_evidence in clause:
        return (True, 1.0, "exact")
    
    # 分句
    sentences = [s.strip() for s in clause.replace("。", "。\n").split("\n") if s.strip()]
    if not sentences:
        sentences = [clause]
    
    # 2. Stage 1: BGE-M3 Dense 召回 Top-K 候选
    candidates = sentences[:3]  # 默认取前3句
    if embedding_model is not None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            evidence_embedding = embedding_model.encode([clean_evidence])
            sentence_embeddings = embedding_model.encode(sentences)
            
            similarities = cosine_similarity(evidence_embedding, sentence_embeddings)[0]
            
            # 取 Top-3 候选进入精排
            top_k_indices = similarities.argsort()[-3:][::-1]
            candidates = [sentences[i] for i in top_k_indices]
            
            # 如果没有 reranker，直接使用 embedding 相似度
            if reranker is None:
                max_similarity = float(max(similarities))
                if max_similarity >= threshold:
                    return (True, max_similarity, "semantic")
        except Exception as e:
            logging.warning(f"语义相似度计算失败: {e}")
    
    # 3. Stage 2: Reranker 精排
    if reranker is not None and candidates:
        try:
            rerank_results = reranker.rerank(
                query=clean_evidence,
                candidates=[{"text": s} if isinstance(s, str) else s for s in candidates],
                top_k=1
            )
            if rerank_results:
                best_score = rerank_results[0][1]  # (doc, score)
                if best_score >= threshold:
                    return (True, best_score, "reranker_match")
                else:
                    return (False, best_score, "hallucination")
        except Exception as e:
            logging.warning(f"Reranker 精排失败: {e}")
    
    # 4. 回退：模糊匹配（字符重叠）
    evidence_words = set(clean_evidence)
    clause_words = set(clause)
    overlap = len(evidence_words & clause_words) / len(evidence_words) if evidence_words else 0
    
    if overlap >= 0.6:
        return (True, overlap, "fuzzy")
    
    return (False, overlap, "hallucination")



def verify_law_citation(
    law_reference: str, 
    reference_info: str = "",
    law_db_path: str = None
) -> tuple:
    """验证法律引用是否有效（分级验证）
    
    验证顺序:
    1. 检查是否在 reference_info 中提及
    2. 检查法律名称是否在数据库/白名单中
    3. 检查条款号格式是否正确
    4. 检查引用内容是否与数据库一致（如有）
    
    Args:
        law_reference: LLM 输出的法条引用
        reference_info: 规则引擎提供的参考信息
        law_db_path: 法律数据库路径（可选）
    
    Returns:
        tuple: (is_valid: bool, validation_level: int, detail: str)
        - is_valid: 法条是否有效
        - validation_level: 验证通过的层级（1-4，0表示失败）
        - detail: 验证详情
    """
    import re
    
    if not law_reference or law_reference in ["无", "None", "", "留空"]:
        return (True, 0, "empty")  # 无引用不算幻觉
    
    # 提取法律名称（支持《》和【】两种格式）
    # 格式1: 《劳动合同法》第X条
    # 格式2: 【民法典 第X条】
    law_name_pattern1 = r"[《]([^》]+)[》]"  # 书名号格式
    law_name_pattern2 = r"[【]([^第]+?)[\s第]"  # 方括号格式（提取到"第"字之前）
    law_name_pattern3 = r"(民法典|劳动合同法|劳动法|合同法|物权法|消费者权益保护法|社会保险法|个人信息保护法)"  # 直接匹配常见法律名
    
    law_names_found = re.findall(law_name_pattern1, law_reference)
    law_names_found.extend(re.findall(law_name_pattern2, law_reference))
    law_names_found.extend(re.findall(law_name_pattern3, law_reference))
    # 去重
    law_names_found = list(set([name.strip() for name in law_names_found if name.strip()]))
    
    # 提取条款号
    article_pattern = r"第([一二三四五六七八九十百千零\d]+)条"
    articles_found = re.findall(article_pattern, law_reference)
    
    # ========== 层级1：检查是否在 reference_info 中 ==========
    if reference_info:
        for law_name in law_names_found:
            if law_name in reference_info:
                return (True, 1, f"在 reference_info 中找到: {law_name}")
    
    # ========== 层级2：检查法律名称是否在白名单/数据库中 ==========
    valid_laws = [
        # 劳动法相关
        "劳动合同法", "劳动法", "社会保险法", "工伤保险条例", 
        "劳动争议调解仲裁法", "就业促进法", "职业病防治法",
        "工资支付暂行规定", "带薪年休假条例", "最低工资规定",
        "女职工劳动保护特别规定", "劳动保障监察条例",
        # 民事法相关
        "民法典", "合同法", "物权法", "担保法", "侵权责任法",
        # 消费者保护相关
        "消费者权益保护法", "产品质量法", "电子商务法",
        # 其他
        "个人信息保护法", "数据安全法", "反不正当竞争法",
        "公司法", "保险法", "招标投标法", "政府采购法",
    ]
    
    law_name_valid = False
    for law_name in law_names_found:
        # 去除"中华人民共和国"前缀
        clean_name = law_name.replace("中华人民共和国", "")
        if clean_name in valid_laws or any(v in clean_name for v in valid_laws):
            law_name_valid = True
            break
    
    if law_name_valid:
        # ========== 层级3：检查条款号格式 ==========
        if articles_found:
            # 验证条款号是否合理（1-999条）
            for article in articles_found:
                try:
                    # 转换中文数字
                    cn_to_num = {
                        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
                        "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
                        "二十": 20, "三十": 30, "四十": 40, "五十": 50,
                        "六十": 60, "七十": 70, "八十": 80, "九十": 90,
                        "一百": 100
                    }
                    if article in cn_to_num:
                        article_num = cn_to_num[article]
                    elif article.isdigit():
                        article_num = int(article)
                    else:
                        # 复杂中文数字（如"二十三"）暂时跳过验证
                        return (True, 3, f"条款号格式正确: 第{article}条")
                    
                    if 1 <= article_num <= 999:
                        return (True, 3, f"条款号格式正确: 第{article}条")
                except:
                    pass
            
            return (True, 3, "条款号格式正确（未验证数值）")
        else:
            return (True, 2, f"法律名称有效: {law_names_found}")
    
    # ========== 层级4：内容验证（需要法律数据库）==========
    # 暂未实现，预留接口
    
    # 验证失败
    if law_names_found:
        return (False, 0, f"未知法律名称: {law_names_found}")
    else:
        return (False, 0, f"无法识别法律引用: {law_reference[:50]}...")


# ============================================================================
# 评测执行器
# ============================================================================

class AblationBenchmark:
    """消融实验评测器"""
    
    def __init__(self, mode: int, source: str = "local"):
        self.mode = mode
        self.config = load_config()
        
        # 读取消融实验配置（复用混合检索和 Reranker 的共享参数）
        ablation_cfg = self.config.get("ablation_benchmark_config", {})
        hallucination_cfg = ablation_cfg.get("hallucination_detection", {})
        reranker_cfg = self.config.get("reranker_config", {})
        hybrid_cfg = self.config.get("hybrid_search_config", {})
        
        # 幻觉检测阈值：复用 reranker_config.threshold
        self.hallucination_threshold = reranker_cfg.get("threshold", 0.3)
        # 召回 Top-K：复用 hybrid_search_config.top_k
        self.recall_top_k = hybrid_cfg.get("top_k", 3)
        # 是否使用两阶段检测
        self.use_two_stage = hallucination_cfg.get("use_two_stage", True)
        
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
        
        # 初始化 Embedding 模型（用于语义相似度评估，替代 LLM-as-a-Judge）
        self.embedding_model = None
        try:
            from sentence_transformers import SentenceTransformer
            embedding_cfg = self.config.get("embedding_config", {})
            model_path = embedding_cfg.get("model_path", "BAAI/bge-small-zh-v1.5")
            self.embedding_model = SentenceTransformer(model_path)
            print(f"✅ Embedding 模型加载成功: {model_path}")
        except Exception as e:
            print(f"⚠️ Embedding 模型加载失败，将仅使用关键词匹配: {e}")
        
        # 初始化 Reranker 模型（用于两阶段幻觉检测）
        self.reranker = None
        if self.use_two_stage:
            try:
                from src.core.reranker import get_reranker
                self.reranker = get_reranker()
                print(f"✅ Reranker 模型加载成功 (幻觉检测阈值: {self.hallucination_threshold})")
            except Exception as e:
                print(f"⚠️ Reranker 模型加载失败，将仅使用 Embedding 相似度: {e}")
    
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
    
    def get_reference_info(self, clause: str) -> tuple:
        """获取规则引擎的参考信息（使用统一检索器）
        
        Returns:
            tuple: (reference_info, law_contents, risk_ids, scores)
        """
        if self.rule_engine is None:
            return "", [], [], []
        
        try:
            # 使用统一的检索器模块
            from src.core.reference_retriever import retrieve_reference
            result = retrieve_reference(clause)
            return result.reference_info, result.law_contents, result.risk_ids, result.scores
        except Exception as e:
            print(f"Reference retrieval error: {e}")
        
        return "", [], [], []
    
    async def analyze_clause(self, clause: str) -> tuple:
        """分析单个条款
        
        Returns:
            tuple: (ParsedResult, reference_info, risk_ids, scores, reflection_info)
        """
        # 获取参考信息（模式3和4使用 Top-K 检索）
        reference_info, law_contents, risk_ids, scores = self.get_reference_info(clause)
        
        # 构建Prompt
        prompt = self.get_prompt(clause, reference_info)
        
        reflection_info = None  # 自反思结果
        
        try:
            # 使用 OllamaClient 的 achat 方法
            content = await self.llm.achat(prompt)
            
            # 解析输出
            result = parse_markdown_output(content)
            
            # 注入法条内容（如果有）
            if law_contents and result.law_reference == "":
                result.law_reference = law_contents[0] if law_contents[0] else ""
            
            # ========== 自反思机制（仅 Mode 3 和 Mode 4）==========
            if self.mode in [EvalMode.CURRENT_WORKFLOW, EvalMode.OPTIMIZED_WORKFLOW]:
                result, reflection_info = await self._apply_self_reflection(clause, result)
            
            return result, reference_info, risk_ids, scores, reflection_info
            
        except Exception as e:
            print(f"LLM error: {e}")
            return ParsedResult(), reference_info, risk_ids, scores, None
    
    async def _apply_self_reflection(self, clause: str, initial_result: ParsedResult) -> tuple:
        """应用自反思机制
        
        Args:
            clause: 条款原文
            initial_result: 初次分析结果
        
        Returns:
            tuple: (调整后的 ParsedResult, reflection_info dict)
        """
        reflection_info = {
            "applied": True,
            "initial_level": initial_result.risk_level,
            "final_level": initial_result.risk_level,
            "adjusted": False,
            "reason": ""
        }
        
        # 构建自反思 Prompt
        reflection_prompt = SELF_REFLECTION_PROMPT.format(
            clause_text=clause,
            risk_level=initial_result.risk_level or "未知",
            risk_reason=initial_result.risk_name or "",
            evidence=initial_result.evidence or "无",
            analysis=initial_result.analysis or ""
        )
        
        try:
            # 调用 LLM 进行反思
            reflection_content = await self.llm.achat(reflection_prompt)
            
            # 解析反思结果
            reflection_result = parse_reflection_output(reflection_content)
            reflection_info["reason"] = reflection_result.get("reason", "")
            
            # 如果结论是"调级"，且有新的等级
            if reflection_result.get("conclusion") == "调级" and reflection_result.get("new_level"):
                new_level = reflection_result["new_level"]
                reflection_info["final_level"] = new_level
                reflection_info["adjusted"] = True
                
                # 更新 ParsedResult 的风险等级
                initial_result.risk_level = new_level
                
                # 在分析中添加调级说明
                adjustment_note = f"\n[二审调级: {reflection_info['initial_level']}→{new_level}，理由: {reflection_info['reason']}]"
                initial_result.analysis = (initial_result.analysis or "") + adjustment_note
                
                print(f"  🔄 自反思调级: {reflection_info['initial_level']} → {new_level}")
            else:
                print(f"  ✓ 自反思维持: {initial_result.risk_level}")
                
        except Exception as e:
            print(f"  ⚠️ 自反思失败: {e}")
            reflection_info["applied"] = False
        
        return initial_result, reflection_info
    
    def evaluate_reason(self, clause: str, gt_keywords: List[str], ai_reason: str) -> float:
        """使用算法+语义匹配评估论证质量（替代 LLM-as-a-Judge）
        
        评分方法：
        1. 关键词匹配（60%权重）：检查 AI 理由是否包含 ground_truth 关键词
        2. 语义相似度（40%权重）：使用 Embedding 计算向量余弦相似度
        
        Returns:
            float: 0.0-1.0 的匹配分数
        """
        if not ai_reason or not gt_keywords:
            return 0.0
        
        ai_reason_lower = ai_reason.lower()
        
        # 1. 关键词精确匹配
        matched_keywords = sum(1 for kw in gt_keywords if kw.lower() in ai_reason_lower)
        keyword_ratio = matched_keywords / len(gt_keywords) if gt_keywords else 0
        
        # 2. 语义相似度匹配（使用 Embedding）
        semantic_sim = 0.0
        if self.embedding_model:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                gt_text = " ".join(gt_keywords)
                embeddings = self.embedding_model.encode([gt_text, ai_reason])
                semantic_sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            except Exception as e:
                print(f"语义相似度计算失败: {e}")
                semantic_sim = 0.0
        
        # 3. 综合评分（加权求和）
        alpha = 0.6  # 关键词匹配权重
        score = alpha * keyword_ratio + (1 - alpha) * semantic_sim
        
        return score
    
    async def evaluate_single(self, item: Dict[str, Any], metrics: EvalMetrics) -> Dict[str, Any]:
        """评估单个样本"""
        import time
        
        text = item.get("text", "")
        gt = item.get("ground_truth", {})
        original_data = item.get("original_data", {})  # LLM 数据集的原始数据
        
        # 记录开始时间
        start_time = time.time()
        
        # 分析条款（返回元组：result, reference_info, risk_ids, scores, reflection_info）
        result, reference_info, matched_risk_ids, matched_scores, reflection_info = await self.analyze_clause(text)
        
        # 记录响应时间
        latency = time.time() - start_time
        metrics.total_latency += latency
        
        # 自反思统计
        if reflection_info and reflection_info.get("applied"):
            metrics.reflection_calls += 1
            if reflection_info.get("adjusted"):
                metrics.reflection_adjustments += 1
            else:
                metrics.reflection_maintain += 1
        
        metrics.total += 1
        
        # 解析成功率
        if result.parse_success:
            metrics.parse_success += 1
        
        # 风险等级评估（支持高/中/低三级）
        gt_risk = gt.get("risk_level", "")
        pred_risk = result.risk_level
        
        # 精确匹配或兼容匹配
        is_risk_correct = (gt_risk == pred_risk) or (gt_risk in pred_risk)
        
        if is_risk_correct:
            metrics.correct_risk += 1
        
        # 加权评分（精确匹配1分，差一级0.5分，差两级0分）
        weighted_score = EvalMetrics.calculate_weighted_score(gt_risk, pred_risk)
        metrics.total_weighted_score += weighted_score
        
        # 混淆矩阵更新（基于"有风险"(高/中) vs "无风险"(低)的二分类）
        # 高/中风险视为正例(Positive)，低风险视为负例(Negative)
        gt_is_risky = gt_risk in ["高", "中"]
        pred_is_risky = pred_risk in ["高", "中"]
        
        if pred_is_risky and gt_is_risky:
            metrics.true_positive += 1
        elif pred_is_risky and not gt_is_risky:
            metrics.false_positive += 1
        elif not pred_is_risky and gt_is_risky:
            metrics.false_negative += 1
        else:
            metrics.true_negative += 1
            
        # [新增] 更新三分类混淆矩阵
        metrics.update_confusion_matrix(gt_risk, pred_risk)
        
        # risk_id 匹配评估（针对 LLM 生成的数据集）
        expected_risks = original_data.get("expected_risks", [])
        if expected_risks:
            # 有预期风险，检查是否正确识别
            metrics.risk_id_total += 1
            # 如果预测为有风险且样本确实包含风险，算匹配成功
            if pred_is_risky and gt_is_risky:
                metrics.risk_id_match += 1
        
        # ===== 方法一：证据一致性评估（两阶段：BGE-M3 + Reranker）=====
        # 合同条款证据验证
        evidence_result = verify_evidence(
            result.evidence, text, 
            embedding_model=self.embedding_model,
            reranker=self.reranker,
            threshold=self.hallucination_threshold
        )
        clause_evidence_valid = evidence_result[0]  # is_valid
        evidence_similarity = evidence_result[1]    # similarity_score
        evidence_match_type = evidence_result[2]    # match_type
        
        if clause_evidence_valid:
            metrics.clause_evidence_valid += 1
            metrics.evidence_valid += 1  # 保持兼容
        else:
            metrics.clause_evidence_invalid += 1
            metrics.evidence_invalid += 1  # 保持兼容
        
        # 1b. 法律引用验证（分级验证：reference_info → 白名单 → 条款号）
        # 注意：低风险样本不需要法律引用，跳过验证
        if gt_risk == "低":
            # 低风险样本法律引用默认有效
            law_citation_valid = True
            law_validation_level = 0
            law_validation_detail = "低风险样本无需法律引用"
            metrics.law_citation_valid += 1
        else:
            # 高/中风险样本需要验证法律引用
            law_result = verify_law_citation(result.law_reference, reference_info=reference_info)
            law_citation_valid = law_result[0]       # is_valid
            law_validation_level = law_result[1]     # validation_level
            law_validation_detail = law_result[2]    # detail
            
            if law_citation_valid:
                metrics.law_citation_valid += 1
            else:
                metrics.law_citation_invalid += 1
        
        # ===== 方法二：规则触发一致性（RAG 检索器评测）=====
        # 从测试用例获取应触发的规则 ID
        expected_rule_ids = set()
        if expected_risks:
            for risk in expected_risks:
                if isinstance(risk, dict) and risk.get("risk_id"):
                    expected_rule_ids.add(risk.get("risk_id"))
        
        # 实际触发的规则 ID（来自检索器 ReferenceResult.risk_ids）
        triggered_rule_ids = set(matched_risk_ids) if matched_risk_ids else set()
        
        # 计算指标（精确字符串匹配）
        if expected_rule_ids:
            metrics.rule_target_count += len(expected_rule_ids)          # 应触发
            metrics.rule_trigger_count += len(triggered_rule_ids)        # 实际触发
            correct_triggers = expected_rule_ids & triggered_rule_ids    # 交集
            metrics.rule_correct_count += len(correct_triggers)          # 正确触发

        
        # ===== 方法三：任务成功率评估 =====
        # 任务成功需同时满足：解析成功 + 风险等级正确(±1级) + 证据有效 + 有建议
        has_suggestion = bool(result.suggestion) and result.suggestion not in ["无", "None", ""]
        is_risk_acceptable = weighted_score >= 0.5  # 精确匹配或差一级都算可接受
        
        if result.parse_success and is_risk_acceptable and clause_evidence_valid and has_suggestion:
            metrics.task_success_count += 1
        
        # 论证质量评估（如果有 reason_keywords）
        reason_keywords = gt.get("reason_keywords", [])
        if reason_keywords:
            reason_score = self.evaluate_reason(text, reason_keywords, result.analysis)
            if reason_score >= 0.5:  # 使用阈值判断匹配成功
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
            "weighted_score": weighted_score,
            # 证据验证详情
            "evidence_valid": clause_evidence_valid,
            "evidence_similarity": round(evidence_similarity, 3),
            "evidence_match_type": evidence_match_type,
            # 法条验证详情
            "law_citation_valid": law_citation_valid,
            "law_validation_level": law_validation_level,
            "law_validation_detail": law_validation_detail,
            # 任务成功
            "task_success": result.parse_success and is_risk_acceptable and clause_evidence_valid and has_suggestion,
        }


def convert_llm_dataset_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 LLM 生成的数据集格式转换为 benchmark 期望的格式
    
    新格式 (399样本):
    {
        "id": "LABOR_001_high_positive_1",
        "contract_text": "...",
        "expected_risks": [{"risk_id": "...", "risk_level": "high", ...}],
        "case_type": "high_positive/medium_positive/negative",
        "source_domain": "LABOR"
    }
    
    旧格式 (兼容):
    {
        "id": "GENERAL_001_pos_1",
        "contract_text": "...",
        "expected_risks": [...],
        "case_type": "positive/negative/boundary"
    }
    
    Benchmark 格式:
    {
        "id": "...",
        "text": "...",
        "ground_truth": {"risk_level": "高/中/低", "reason_keywords": [...]},
        "original_data": {...}  # 保留原始数据用于 risk_id 匹配
    }
    """
    # 检测是否为 LLM 数据集格式
    if "contract_text" in item:
        expected_risks = item.get("expected_risks", [])
        case_type = item.get("case_type", "")
        
        # 根据 case_type 确定风险等级
        if case_type == "high_positive":
            risk_level = "高"
        elif case_type == "medium_positive":
            risk_level = "中"
        else:  # negative 或其他
            risk_level = "低"
        
        # 提取关键词作为 reason_keywords
        reason_keywords = []
        for risk in expected_risks:
            if isinstance(risk, dict) and risk.get("risk_name"):
                reason_keywords.append(risk["risk_name"])
        
        return {
            "id": item.get("id", ""),
            "text": item.get("contract_text", ""),
            "ground_truth": {
                "risk_level": risk_level,
                "reason_keywords": reason_keywords,
            },
            "original_data": item,  # 保留原始数据用于 risk_id 匹配
            "source_domain": item.get("source_domain", ""),  # 保留 source_domain
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
            content = f.read().strip()
            
            # 检测文件格式：JSON 数组 vs JSONL
            if content.startswith("["):
                # JSON 数组格式
                raw_items = json.loads(content)
                for raw_item in raw_items:
                    converted_item = convert_llm_dataset_item(raw_item)
                    dataset.append(converted_item)
            else:
                # JSONL 格式（逐行解析）
                for line in content.split("\n"):
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
    
    # 统计数据集信息（高/中/低三级风险）
    high_count = sum(1 for d in dataset if d.get("ground_truth", {}).get("risk_level") == "高")
    medium_count = sum(1 for d in dataset if d.get("ground_truth", {}).get("risk_level") == "中")
    low_count = sum(1 for d in dataset if d.get("ground_truth", {}).get("risk_level") == "低")
    log(f"📊 高风险: {high_count}, 中风险: {medium_count}, 低风险: {low_count}")
    
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
    with open(data_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
        # 检测文件格式：JSON 数组 vs JSONL
        if content.startswith("["):
            # JSON 数组格式
            raw_items = json.loads(content)
        else:
            # JSONL 格式（逐行解析）
            raw_items = [json.loads(line) for line in content.split("\n") if line.strip()]
        
        for raw_item in raw_items:
            converted_item = convert_llm_dataset_item(raw_item)
            dataset.append(converted_item)
    
    total_samples = len(dataset)
    if limit and limit < total_samples:
        # 使用分层采样，确保 High/Medium/Low 均衡 (1:1:1)
        dataset = stratified_sample(dataset, limit)
        print(f"📊 使用分层采样 (Stratified Sampling) 抽取 {limit} 条（High/Medium/Low 1:1:1）")
    
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
    
    # 指标对比 (包含新增的评估指标)
    metric_keys = [
        "accuracy", "weighted_accuracy", "f1", "precision", "recall", "parse_rate",
        "hallucination_rate", "clause_hallucination_rate", "law_hallucination_rate",
        "rule_recall", "rule_precision", "task_success_rate", "avg_latency_sec"
    ]
    for key in metric_keys:
        print(f"{key:<25}", end="")
        for mode in modes:
            mode_key = f"mode_{mode}"
            if mode_key in all_results:
                value = all_results[mode_key]["metrics"].get(key, 0)
                # latency 使用秒数格式
                if "latency" in key:
                    print(f"{value:.3f}s".ljust(15), end="")
                else:
                    print(f"{value:.2%}".ljust(15), end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()
    
    # ========== 创建独立输出目录 ==========
    script_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / f"results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存结果 JSON
    output_path = output_dir / "ablation_results.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存至: {output_dir}")
    
    # ========== 生成两张图表（基础指标 + 高级指标）==========    # 生成图表
    try:
        from evaluation.chart_generator import generate_report_charts
        
        # 结果保存目录
        output_dir = Path(output_path).parent
        
        # 生成图表
        chart_paths = generate_report_charts(all_results, output_dir, timestamp)
        all_results["chart_paths"] = chart_paths
        all_results["timestamp"] = timestamp
        
        # 更新保存的 JSON（包含图表路径）
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"📊 已生成 {len(chart_paths)} 张图表")
    except ImportError as e:
        print(f"⚠️ 图表生成失败（请确保安装 matplotlib）: {e}")
    except Exception as e:
        print(f"⚠️ 图表生成异常: {e}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="消融实验评测脚本")
    parser.add_argument("--data", type=str, default="evaluation/llm_benchmark_dataset.json",
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
