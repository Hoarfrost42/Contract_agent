import jieba
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.utils.config_loader import load_config

class HybridSearcher:
    def __init__(self, model_path: str):
        """
        初始化混合检索器
        :param model_path: Embedding 模型路径
        """
        print(f"Loading Embedding Model from: {model_path} ...")
        self.encoder = SentenceTransformer(model_path)
        self.rules: List[Dict] = []
        self.bm25 = None
        self.rule_embeddings = None
        
        # 从配置文件加载检索参数
        config = load_config()
        search_config = config.get("hybrid_search_config", {})
        self.default_alpha = search_config.get("alpha", 0.7)
        self.threshold = search_config.get("threshold", 0.4)
        
    def build_index(self, rules: List[Dict]):
        """
        构建索引
        :param rules: 规则列表
        """
        self.rules = rules
        if not rules:
            return

        # 1. 准备语料
        corpus = []
        for rule in rules:
            # 组合关键字段用于索引
            text = f"{rule.get('risk_name', '')} {rule.get('analysis_logic', '')} {' '.join(rule.get('keywords', []))}"
            corpus.append(text)
            
        # 2. 构建 Dense Index (Embedding)
        print("Building Dense Index...")
        self.rule_embeddings = self.encoder.encode(corpus, normalize_embeddings=True)
        
        # 3. 构建 Sparse Index (BM25)
        print("Building Sparse Index...")
        tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def search(self, query: str, top_k: int = 1, alpha: float = None, allowed_indices: Optional[List[int]] = None):
        """
        混合检索
        :param query: 查询文本 (条款原文)
        :param top_k: 返回结果数量，1=单结果模式，>1=多结果模式
        :param alpha: 混合权重 (Dense Score * alpha + Sparse Score * (1-alpha))
        :param allowed_indices: 允许检索的规则索引列表
        :return: 
            - top_k=1: (最佳匹配的规则, 匹配分数) 元组
            - top_k>1: (规则列表, 分数列表) 元组，只包含超过阈值的结果
        """
        if not self.rules:
            return None, 0.0
        
        # 使用配置中的默认 alpha 值
        if alpha is None:
            alpha = self.default_alpha
            
        # 1. Dense Search
        query_embedding = self.encoder.encode(query, normalize_embeddings=True)
        # Cosine Similarity
        dense_scores = np.dot(self.rule_embeddings, query_embedding)
        
        # 2. Sparse Search
        tokenized_query = list(jieba.cut(query))
        sparse_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # 归一化 Sparse Scores (Min-Max Normalization)
        if sparse_scores.max() > sparse_scores.min():
            sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min())
        else:
            sparse_scores = np.zeros_like(sparse_scores)
            
        # 3. Hybrid Scoring
        hybrid_scores = alpha * dense_scores + (1 - alpha) * sparse_scores
        
        # 4. Apply Filter
        if allowed_indices is not None:
            # Create a mask initialized to -inf
            mask = np.full_like(hybrid_scores, -np.inf)
            # Set allowed indices to 0 (so adding them keeps original score, effectively)
            # Actually, better to just set disallowed to -inf
            # Let's do: create a full -inf array, then copy allowed scores
            filtered_scores = np.full_like(hybrid_scores, -np.inf)
            filtered_scores[allowed_indices] = hybrid_scores[allowed_indices]
            hybrid_scores = filtered_scores
        
        # 5. Get Top-K
        if top_k == 1:
            # 单结果模式（保持向后兼容）
            best_idx = np.argmax(hybrid_scores)
            best_score = float(hybrid_scores[best_idx])
            
            if best_score == -np.inf:
                return None, 0.0
            
            if best_score < self.threshold: 
                return None, best_score
                
            return self.rules[best_idx], best_score
        else:
            # Top-K 多结果模式
            sorted_indices = np.argsort(hybrid_scores)[::-1]  # 降序排序
            
            results = []
            scores = []
            
            for idx in sorted_indices[:top_k]:
                score = float(hybrid_scores[idx])
                # 只返回超过阈值的结果
                if score >= self.threshold and score != -np.inf:
                    results.append(self.rules[idx])
                    scores.append(score)
            
            return results, scores
