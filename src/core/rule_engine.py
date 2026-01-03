import json
import os
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from src.database.db_manager import DBManager
from src.utils.text_utils import chinese_numeral_to_str
from src.utils.config_loader import load_config
from src.core.hybrid_searcher import HybridSearcher

# 高危关键词列表（用于兜底检测）
HIGH_RISK_KEYWORDS = [
    # 单方权利
    r"甲方有权.{0,10}(解除|终止|变更|修改)",
    r"(单方|一方).{0,5}(决定|有权|可以)",
    r"不得.{0,5}(异议|拒绝|反对)",
    # 责任转嫁
    r"(全部|一切|所有).{0,5}(责任|风险|损失).{0,5}(由|归).{0,5}(乙方|承担)",
    r"无论.{0,10}(原因|情况).{0,10}(承担|负责)",
    # 权利限制
    r"(放弃|让与|转让).{0,5}(权利|权益)",
    r"最终.{0,5}(解释权|裁决)",
    # 违约金
    r"违约金.{0,10}(由|根据).{0,10}(甲方|对方).{0,10}(确定|酌情)",
]

class RuleEngine:
    def __init__(self, rules_path: str = "src/core/risk_rules.json"):
        self.rules = self._load_rules(rules_path)
        self.db_manager = DBManager()
        
        # Build ID to Index Map
        self.id_to_index = {r['risk_id']: i for i, r in enumerate(self.rules)}
        
        # Initialize Hybrid Searcher
        config = load_config()
        model_path = config.get("embedding_config", {}).get("model_path")
        if model_path:
            self.searcher = HybridSearcher(model_path)
            self.searcher.build_index(self.rules)
        else:
            print("Warning: No embedding model path configured. Hybrid search disabled.")
            self.searcher = None

    def _load_rules(self, rules_path: str) -> List[Dict]:
        """从 JSON 文件加载规则库。"""
        # 解析相对于项目根目录的绝对路径
        # 假设此文件在 src/core/，所以项目根目录是 ../../
        base_path = Path(__file__).resolve().parents[2]
        full_path = base_path / rules_path
        
        if not full_path.exists():
            print(f"警告: 规则文件未找到于 {full_path}")
            return []

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载规则出错: {e}")
            return []

    def _keyword_fallback_check(self, clause_text: str) -> Tuple[bool, Optional[str]]:
        """
        关键词兜底检测：当规则库未匹配时，检查是否包含高危关键词。
        返回: (是否触发, 触发的关键词)
        """
        for pattern in HIGH_RISK_KEYWORDS:
            match = re.search(pattern, clause_text)
            if match:
                return True, match.group(0)
        return False, None

    def match_risk(self, clause_text: str, contract_type: str = "通用") -> Tuple[Optional[Dict], float, str]:
        """
        基于混合检索匹配条款与风险规则。
        
        返回: (匹配的规则, 置信度分数, 匹配来源)
        - 匹配来源: "rule_match" | "keyword_fallback" | "no_match"
        """
        from src.core.preprocessor import preprocess_clause
        
        # 1. Preprocessing (Filtering)
        # 传入 contract_type 进行领域过滤
        allowed_rules = preprocess_clause(clause_text, self.rules, contract_type)
        
        if not allowed_rules:
            # 规则库预处理过滤掉了，尝试关键词兜底
            triggered, keyword = self._keyword_fallback_check(clause_text)
            if triggered:
                return None, 0.3, f"keyword_fallback:{keyword}"
            return None, 0.0, "no_match"
            
        allowed_indices = [self.id_to_index[r['risk_id']] for r in allowed_rules if r.get('risk_id') in self.id_to_index]
        
        if not allowed_indices:
            triggered, keyword = self._keyword_fallback_check(clause_text)
            if triggered:
                return None, 0.3, f"keyword_fallback:{keyword}"
            return None, 0.0, "no_match"

        if not clause_text or not self.searcher:
            # Fallback to keyword matching if searcher is not available
            if not clause_text: 
                return None, 0.0, "no_match"
            # Only search within allowed rules
            for rule in allowed_rules:
                keywords = rule.get("keywords", [])
                for keyword in keywords:
                    if keyword in clause_text:
                        return rule, 0.8, "rule_match"
            
            # 关键词兜底
            triggered, keyword = self._keyword_fallback_check(clause_text)
            if triggered:
                return None, 0.3, f"keyword_fallback:{keyword}"
            return None, 0.0, "no_match"
        
        # 混合检索
        rule, score = self.searcher.search(clause_text, allowed_indices=allowed_indices)
        
        if rule:
            return rule, score, "rule_match"
        
        # 未匹配到规则，尝试关键词兜底
        triggered, keyword = self._keyword_fallback_check(clause_text)
        if triggered:
            return None, max(score, 0.3), f"keyword_fallback:{keyword}"
        
        return None, score, "no_match"

    def _search_law(self, laws_str: str) -> str:
        """
        根据 risk_rules.json 中的 laws 字段搜索数据库。
        格式如: "《劳动合同法》第十九条" 或 "《劳动合同法》第二十三条、第二十四条"
        """
        if not laws_str:
            return ""
            
        results = []
        
        # 匹配 《法律名称》条款号
        # 考虑到可能有多个法律或多个条款，这里做一个简单的解析
        # 假设格式主要是 《Law》Article
        
        matches = re.finditer(r"《(.*?)》([^《》]+)", laws_str)
        found_any = False
        
        for match in matches:
            law_name = match.group(1)
            articles_str = match.group(2)
            
            # 分割多个条款，假设用顿号或空格或逗号分隔
            article_ids = re.split(r"[、，, ]+", articles_str)
            
            for art_id in article_ids:
                art_id = art_id.strip()
                if not art_id:
                    continue
                    
                # 尝试精确查询
                # art_id 可能是 "第十九条" 或 "十九条" 或 "19"
                # 需要转换为阿拉伯数字字符串
                query_id = art_id
                try:
                    # 移除 "第" 和 "条"
                    clean_id = art_id.replace("第", "").replace("条", "")
                    query_id = chinese_numeral_to_str(clean_id)
                except ValueError:
                    # 转换失败则保持原样 (可能是纯数字或无法识别的格式)
                    pass
                
                record = self.db_manager.fetch_article(law_name, query_id)
                if record:
                    results.append(f"【{law_name} {art_id}】{record['content']}")
                    found_any = True
        
        if not found_any:
             # 如果正则解析失败或未找到，尝试用整个字符串作为 tag 搜索
             content = self.db_manager.fetch_by_tag(laws_str)
             if content:
                 results.append(f"【相关法条】{content}")

        return "\n".join(results)

    def get_reference_info(self, clause_text: str, contract_type: str = "通用") -> Tuple[str, Optional[str], Optional[str], float, str]:
        """
        获取条款的格式化参考信息。
        
        返回: (适合注入 LLM Prompt 的字符串, 检索到的法律原文内容, 风险规则ID, 置信度分数, 匹配来源)
        """
        rule, confidence, match_source = self.match_risk(clause_text, contract_type)
        
        # 关键词兜底触发
        if match_source.startswith("keyword_fallback:"):
            keyword = match_source.split(":", 1)[1]
            info = (
                f"【⚠️ 关键词预警】检测到高危关键词：\"{keyword}\"\n"
                f"【提示】未匹配到规则库，但条款包含典型风险特征，建议人工复核。\n"
                f"【置信度】{confidence:.0%}（较低，建议谨慎判断）"
            )
            return info, None, None, confidence, match_source
        
        # 未匹配
        if not rule:
            if confidence > 0.2:
                # 有一定分数但未超过阈值
                info = (
                    f"无匹配的专家规则库信息。\n"
                    f"【置信度】{confidence:.0%}（低于阈值，可能为低风险）"
                )
            else:
                info = "无匹配的专家规则库信息。"
            return info, None, None, confidence, match_source
            
        # 正常匹配
        law_content = ""
        laws_str = rule.get('laws', '')
        if laws_str:
            law_content = self._search_law(laws_str)
        
        confidence_label = "高" if confidence >= 0.6 else "中" if confidence >= 0.4 else "低"
        info = (
            f"【匹配规则】{rule.get('risk_name', '未知风险')}\n"
            f"【专家逻辑】{rule.get('analysis_logic', '')}\n"
            f"【法律标签】{rule.get('law_tag', '')}\n"
            f"【置信度】{confidence:.0%}（{confidence_label}）"
        )
        
        if law_content:
            info += f"\n【法规原文】\n{law_content}"
            
        return info, law_content, rule.get('risk_id'), confidence, match_source

    def get_reference_info_topk(self, clause_text: str, top_k: int = 3, contract_type: str = "通用") -> Tuple[str, List[str], List[str], List[float]]:
        """
        获取条款的 Top-K 格式化参考信息（多规则匹配）。
        
        Args:
            clause_text: 条款原文
            top_k: 返回的规则数量
            contract_type: 合同类型（用于过滤规则）
        
        Returns: 
            (格式化的参考信息字符串, 法条内容列表, 风险ID列表, 置信度列表)
        """
        from src.core.preprocessor import preprocess_clause
        
        # 1. Preprocessing
        # 传入 contract_type 进行领域过滤
        allowed_rules = preprocess_clause(clause_text, self.rules, contract_type)
        if not allowed_rules:
            return "无匹配的专家规则库信息。", [], [], []
        
        allowed_indices = [self.id_to_index[r['risk_id']] for r in allowed_rules if r.get('risk_id') in self.id_to_index]
        
        if not allowed_indices or not self.searcher:
            return "无匹配的专家规则库信息。", [], [], []
        
        # 2. Top-K 检索
        rules, scores = self.searcher.search(clause_text, top_k=top_k, allowed_indices=allowed_indices)
        
        if not rules:
            return "无匹配的专家规则库信息。", [], [], []
        
        # 3. 格式化输出
        info_parts = []
        law_contents = []
        risk_ids = []
        
        for i, (rule, score) in enumerate(zip(rules, scores)):
            confidence_label = "高" if score >= 0.6 else "中" if score >= 0.4 else "低"
            
            rule_info = (
                f"--- 候选规则 {i+1} (置信度: {score:.0%} {confidence_label}) ---\n"
                f"【匹配规则】{rule.get('risk_name', '未知风险')}\n"
                f"【专家逻辑】{rule.get('analysis_logic', '')}\n"
                f"【法律标签】{rule.get('law_tag', '')}"
            )
            
            # 检索法条
            laws_str = rule.get('laws', '')
            if laws_str:
                law_content = self._search_law(laws_str)
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
        
        # 添加提示
        if len(rules) > 1:
            combined_info = f"⚠️ 检测到 {len(rules)} 个可能相关的风险点，请逐一评估：\n\n" + combined_info
        
        return combined_info, law_contents, risk_ids, scores

