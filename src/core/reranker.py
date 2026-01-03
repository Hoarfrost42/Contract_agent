"""
BGE Reranker 模块

使用 BAAI/bge-reranker-v2-m3 对检索结果进行重排序，
提升 reference_info 的相关性质量。

流程：
1. 初始检索（Top-K）→ 候选规则列表
2. Rerank → 按相关性重新排序
3. 返回排序后的结果
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RerankedItem:
    """重排序后的单个条目"""
    risk_id: str
    risk_name: str
    original_score: float
    rerank_score: float
    rule_content: dict


class BGEReranker:
    """
    BGE Reranker 重排序器
    
    使用 BAAI/bge-reranker-v2-m3 模型对检索结果进行重排序。
    
    使用方式：
        reranker = BGEReranker()
        reranked = reranker.rerank(query, candidates)
    """
    
    def __init__(self, model_name: str = None, use_fp16: bool = None):
        """
        初始化 Reranker
        
        Args:
            model_name: Hugging Face 模型名称，默认从配置读取
            use_fp16: 是否使用 FP16 加速，默认从配置读取
        """
        from src.utils.config_loader import load_config
        config = load_config()
        reranker_config = config.get("reranker_config", {})
        
        self.model_name = model_name or reranker_config.get("model_name", "BAAI/bge-reranker-v2-m3")
        self.use_fp16 = use_fp16 if use_fp16 is not None else reranker_config.get("use_fp16", True)
        self.reranker = None
        self._load_model()
    
    def _load_model(self):
        """加载 Reranker 模型"""
        try:
            from FlagEmbedding import FlagReranker
            
            print(f"🔄 正在加载 Reranker 模型: {self.model_name} ...")
            self.reranker = FlagReranker(
                self.model_name, 
                use_fp16=self.use_fp16
            )
            print(f"✅ Reranker 模型加载成功")
            
        except ImportError:
            logger.warning("FlagEmbedding 未安装，请运行: pip install FlagEmbedding")
            print("⚠️ FlagEmbedding 未安装，Reranker 功能不可用")
            self.reranker = None
        except Exception as e:
            logger.error(f"Reranker 模型加载失败: {e}")
            print(f"❌ Reranker 模型加载失败: {e}")
            self.reranker = None
    
    def rerank(
        self, 
        query: str, 
        candidates: List[dict],
        top_k: int = 3
    ) -> List[Tuple[dict, float]]:
        """
        对候选规则进行重排序
        
        Args:
            query: 查询文本（条款原文）
            candidates: 候选规则列表，每个规则是一个 dict
            top_k: 返回的结果数量
            
        Returns:
            List[Tuple[dict, float]]: 重排序后的 [(规则, rerank_score), ...]
        """
        if not candidates:
            return []
        
        if self.reranker is None:
            # Reranker 不可用，返回原始顺序
            logger.warning("Reranker 不可用，返回原始顺序")
            return [(c, 0.5) for c in candidates[:top_k]]
        
        try:
            # 构建 query-passage 对
            pairs = []
            for candidate in candidates:
                # 使用规则的 risk_name + analysis_logic 作为 passage
                passage = f"{candidate.get('risk_name', '')}: {candidate.get('analysis_logic', '')}"
                pairs.append([query, passage])
            
            # 计算 rerank 分数
            scores = self.reranker.compute_score(pairs, normalize=True)
            
            # 如果是单个结果，转换为列表
            if isinstance(scores, float):
                scores = [scores]
            
            # 组合并排序
            scored_candidates = list(zip(candidates, scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 返回 Top-K
            return scored_candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Rerank 失败: {e}")
            print(f"⚠️ Rerank 失败: {e}，返回原始顺序")
            return [(c, 0.5) for c in candidates[:top_k]]
    
    def rerank_with_details(
        self,
        query: str,
        candidates: List[dict],
        original_scores: List[float],
        top_k: int = 3
    ) -> List[RerankedItem]:
        """
        带详细信息的重排序
        
        Args:
            query: 查询文本
            candidates: 候选规则列表
            original_scores: 原始检索分数
            top_k: 返回数量
            
        Returns:
            List[RerankedItem]: 重排序后的详细结果
        """
        reranked = self.rerank(query, candidates, top_k=len(candidates))
        
        results = []
        for rule, rerank_score in reranked[:top_k]:
            # 找到原始分数
            idx = candidates.index(rule) if rule in candidates else -1
            orig_score = original_scores[idx] if idx >= 0 and idx < len(original_scores) else 0.0
            
            results.append(RerankedItem(
                risk_id=rule.get("risk_id", ""),
                risk_name=rule.get("risk_name", ""),
                original_score=orig_score,
                rerank_score=rerank_score,
                rule_content=rule
            ))
        
        return results


# ============================================================================
# 单例模式
# ============================================================================

_reranker_instance: Optional[BGEReranker] = None


def get_reranker(model_name: str = None) -> BGEReranker:
    """
    获取 Reranker 单例
    
    Args:
        model_name: 可选，覆盖默认模型名称
        
    Returns:
        BGEReranker 实例（如配置禁用则返回 None）
    """
    from src.utils.config_loader import load_config
    config = load_config()
    reranker_config = config.get("reranker_config", {})
    
    # 检查是否启用 Reranker
    if not reranker_config.get("enabled", True):
        return None
    
    global _reranker_instance
    
    if _reranker_instance is None:
        _reranker_instance = BGEReranker(
            model_name=model_name or reranker_config.get("model_name")
        )
    
    return _reranker_instance


def rerank_candidates(
    query: str, 
    candidates: List[dict], 
    top_k: int = 3
) -> List[Tuple[dict, float]]:
    """
    便捷函数：对候选规则进行重排序
    
    用法：
        from src.core.reranker import rerank_candidates
        reranked = rerank_candidates(clause_text, rules)
    """
    reranker = get_reranker()
    return reranker.rerank(query, candidates, top_k)
