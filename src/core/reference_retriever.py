"""
统一的参考信息检索器模块

提供 Top-K 检索 + Rerank 功能，供以下模块统一调用：
- engine.py (实际工作流)
- ablation_benchmark.py (评测脚本)
- llm.py (如需要)

流程：
1. 初始检索（BM25 + Dense）→ Top-K 候选
2. Rerank（BGE-Reranker）→ 重排序
3. 格式化输出 → reference_info
"""

from typing import Tuple, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReferenceResult:
    """检索结果数据类"""
    reference_info: str          # 格式化的参考信息字符串
    law_contents: List[str]      # 法条内容列表
    risk_ids: List[str]          # 匹配的风险ID列表
    scores: List[float]          # 置信度分数列表（rerank 后）
    match_source: str            # 匹配来源: "reranked" | "topk_match" | "no_match"
    reranked: bool = False       # 是否经过 rerank
    pre_filter_max_score: float = 0.0  # 阈值过滤前的最高分数（用于调试）
    
    @property
    def has_match(self) -> bool:
        """是否有匹配结果"""
        return len(self.risk_ids) > 0
    
    @property
    def best_score(self) -> float:
        """最高置信度"""
        return self.scores[0] if self.scores else 0.0
    
    @property
    def best_risk_id(self) -> Optional[str]:
        """最佳匹配的风险ID"""
        return self.risk_ids[0] if self.risk_ids else None


class ReferenceRetriever:
    """
    统一的参考信息检索器（支持 Rerank）
    
    使用方式：
        retriever = ReferenceRetriever(use_rerank=True)
        result = retriever.retrieve(clause_text)
        print(result.reference_info)
    """
    
    # Rerank 分数阈值：低于此分数的结果将被过滤
    RERANK_THRESHOLD = 0.3
    
    def __init__(self, top_k: int = None, use_rerank: bool = None, rerank_threshold: float = None):
        """
        初始化检索器
        
        Args:
            top_k: 返回的候选规则数量，默认从配置读取
            use_rerank: 是否使用 BGE Reranker 重排序，默认从配置读取
            rerank_threshold: Rerank 分数阈值，默认从配置读取
        """
        from src.core.rule_engine import RuleEngine
        from src.utils.config_loader import load_config
        
        self.rule_engine = RuleEngine()
        
        config = load_config()
        reranker_config = config.get("reranker_config", {})
        
        self.top_k = top_k or config.get("hybrid_search_config", {}).get("top_k", 3)
        self.use_rerank = use_rerank if use_rerank is not None else reranker_config.get("enabled", True)
        self.rerank_threshold = rerank_threshold if rerank_threshold is not None else reranker_config.get("threshold", 0.3)
        
        # 懒加载 Reranker
        self.reranker = None
        if self.use_rerank:
            self._init_reranker()
    
    def _init_reranker(self):
        """初始化 Reranker（懒加载）"""
        try:
            from src.core.reranker import get_reranker
            self.reranker = get_reranker()
        except Exception as e:
            logger.warning(f"Reranker 初始化失败，将使用原始排序: {e}")
            self.reranker = None
    
    def retrieve(self, clause_text: str, contract_type: str = "通用") -> ReferenceResult:
        """
        检索条款的参考信息（Top-K + Rerank 模式）
        
        Args:
            clause_text: 待检索的条款文本
            contract_type: 合同类型（用于过滤规则）
            
        Returns:
            ReferenceResult: 包含格式化参考信息、法条、风险ID和置信度的结果对象
        """
        try:
            from src.core.preprocessor import preprocess_clause
            
            # ========== Step 1: 预处理和初始检索 ==========
            # 传入 contract_type 进行领域过滤
            allowed_rules = preprocess_clause(clause_text, self.rule_engine.rules, contract_type)
            if not allowed_rules:
                return self._empty_result("no_match")
            
            allowed_indices = [
                self.rule_engine.id_to_index[r['risk_id']] 
                for r in allowed_rules 
                if r.get('risk_id') in self.rule_engine.id_to_index
            ]
            
            if not allowed_indices or not self.rule_engine.searcher:
                return self._empty_result("no_match")
            
            # 初始检索：获取更多候选（rerank 前）
            initial_top_k = self.top_k * 2 if self.use_rerank else self.top_k
            rules, scores = self.rule_engine.searcher.search(
                clause_text, 
                top_k=initial_top_k, 
                allowed_indices=allowed_indices
            )
            
            if not rules:
                return self._empty_result("no_match")
            
            # ========== Step 2: Rerank（可选）==========
            reranked = False
            # 改为 >= 1，即使只有1条结果也进行 Rerank 验证分数
            if self.use_rerank and self.reranker is not None and len(rules) >= 1:
                try:
                    reranked_results = self.reranker.rerank(
                        query=clause_text,
                        candidates=rules,
                        top_k=self.top_k * 2  # 获取更多候选，后续按阈值过滤
                    )
                    # 解包重排序结果
                    rules = [r for r, s in reranked_results]
                    scores = [s for r, s in reranked_results]
                    reranked = True
                    print(f"🔄 Rerank 完成: {len(reranked_results)} 条结果")
                except Exception as e:
                    logger.warning(f"Rerank 失败，使用原始排序: {e}")
                    rules = rules[:self.top_k]
                    scores = scores[:self.top_k]
            else:
                # 不使用 rerank，直接截取 top_k
                rules = rules[:self.top_k]
                scores = scores[:self.top_k]
            
            # ========== Step 2.5: 阈值过滤（关键！）==========
            if reranked and self.rerank_threshold > 0:
                # 过滤掉低于阈值的结果
                filtered = [(r, s) for r, s in zip(rules, scores) if s >= self.rerank_threshold]
                
                if not filtered:
                    # 关键：Rerank 后没有通过阈值的结果，给 Prompt 明确信号
                    pre_filter_max = max(scores) if scores else 0.0
                    print(f"⚠️ Rerank 后所有结果分数 < {self.rerank_threshold}，最高分: {pre_filter_max:.2f}，返回空参考信息")
                    return self._empty_result_with_signal(pre_filter_max_score=pre_filter_max)
                
                # 截取 top_k
                filtered = filtered[:self.top_k]
                rules = [r for r, s in filtered]
                scores = [s for r, s in filtered]
                print(f"📊 阈值过滤后保留 {len(rules)} 条结果 (threshold={self.rerank_threshold})")
            
            # ========== Step 3: 格式化输出 ==========
            reference_info, law_contents, risk_ids = self._format_results(rules, scores)
            
            return ReferenceResult(
                reference_info=reference_info,
                law_contents=law_contents,
                risk_ids=risk_ids,
                scores=scores,
                match_source="reranked" if reranked else "topk_match",
                reranked=reranked
            )
            
        except Exception as e:
            logger.error(f"Reference retrieval error: {e}")
            return self._empty_result("error")
    
    def _format_results(
        self, 
        rules: List[dict], 
        scores: List[float]
    ) -> Tuple[str, List[str], List[str]]:
        """格式化检索结果为 reference_info"""
        info_parts = []
        law_contents = []
        risk_ids = []
        
        for i, (rule, score) in enumerate(zip(rules, scores)):
            confidence_label = "高" if score >= 0.6 else "中" if score >= 0.4 else "低"
            
            rule_info = (
                f"--- 候选规则 {i+1} (相关度: {score:.0%} {confidence_label}) ---\n"
                f"【匹配规则】{rule.get('risk_name', '未知风险')}\n"
                f"【专家逻辑】{rule.get('analysis_logic', '')}\n"
                f"【法律标签】{rule.get('law_tag', '')}"
            )
            
            # 检索法条
            laws_str = rule.get('laws', '')
            if laws_str:
                law_content = self.rule_engine._search_law(laws_str)
                if law_content:
                    rule_info += f"\n【法规原文】\n{law_content}"
                    law_contents.append(law_content)
                else:
                    law_contents.append("")
            else:
                law_contents.append("")
            
            info_parts.append(rule_info)
            risk_ids.append(rule.get('risk_id', ''))
        
        combined_info = "\n\n".join(info_parts)
        
        if len(rules) > 1:
            combined_info = f"⚠️ 检测到 {len(rules)} 个可能相关的风险点，请逐一评估：\n\n" + combined_info
        
        return combined_info, law_contents, risk_ids
    
    def _empty_result(self, source: str) -> ReferenceResult:
        """返回空结果"""
        return ReferenceResult(
            reference_info="无匹配的专家规则库信息。",
            law_contents=[],
            risk_ids=[],
            scores=[],
            match_source=source,
            reranked=False
        )
    
    def _empty_result_with_signal(self, pre_filter_max_score: float = 0.0) -> ReferenceResult:
        """
        返回空结果（Rerank 后无高分匹配）
        
        严格返回"无"，让 Prompt 的"空上下文"规则生效。
        保留过滤前的最高分数用于调试。
        """
        return ReferenceResult(
            reference_info="无",
            law_contents=[],
            risk_ids=[],
            scores=[],
            match_source="rerank_filtered",
            reranked=True,
            pre_filter_max_score=pre_filter_max_score
        )
    
    def retrieve_single(self, clause_text: str) -> Tuple[Optional[dict], float, str]:
        """
        检索单个最佳匹配（兼容旧接口）
        
        Returns:
            tuple: (matched_rule, confidence, match_source)
        """
        try:
            matched_rule, confidence, match_source = self.rule_engine.match_risk(clause_text)
            return matched_rule, confidence, match_source
        except Exception as e:
            print(f"Single match error: {e}")
            return None, 0.0, "error"


# ============================================================================
# 模块级单例（懒加载）
# ============================================================================

_retriever_instance: Optional[ReferenceRetriever] = None


def get_retriever(top_k: int = None, use_rerank: bool = True) -> ReferenceRetriever:
    """
    获取检索器单例
    
    Args:
        top_k: 可选，覆盖默认的 top_k 值
        use_rerank: 是否使用 BGE Reranker
        
    Returns:
        ReferenceRetriever 实例
    """
    global _retriever_instance
    
    if _retriever_instance is None:
        _retriever_instance = ReferenceRetriever(top_k=top_k, use_rerank=use_rerank)
    
    return _retriever_instance


def retrieve_reference(clause_text: str, top_k: int = None, use_rerank: bool = True, contract_type: str = "通用") -> ReferenceResult:
    """
    便捷函数：检索条款的参考信息
    
    用法：
        from src.core.reference_retriever import retrieve_reference
        result = retrieve_reference("甲方有权单方解除合同...")
        print(result.reference_info)
        print(f"是否经过 Rerank: {result.reranked}")
    """
    retriever = get_retriever(top_k, use_rerank)
    return retriever.retrieve(clause_text, contract_type)
