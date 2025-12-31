import re
import jieba.posseg as pseg
from typing import List, Dict, Any, Set

class ClauseClassifier:
    """
    条款分类器：根据关键词或正则将条款归类。
    """
    
    CATEGORIES = {
        "definition": [r"定义", r"释义", r"是指", r"含义"],
        "info": [r"地址", r"联系方式", r"账号", r"开户行", r"税号", r"邮编", r"电话", r"传真"],
        "rights_obligations": [r"权利", r"义务", r"有权", r"应", r"必须", r"不得", r"负责"],
        "change": [r"变更", r"修改", r"调整", r"补充"],
        "liability": [r"违约", r"赔偿", r"责任", r"罚款", r"滞纳金"],
        "dispute": [r"争议", r"管辖", r"仲裁", r"诉讼", r"法律适用"],
        "termination": [r"解除", r"终止", r"撤销"]
    }
    
    # 开场白特征模式（这些条款通常是合同头部的程序性内容）
    OPENING_PATTERNS = [
        r"甲方[（(].*?[）)][:：]",  # 甲方（公司名称）：
        r"乙方[（(].*?[）)][:：]",  # 乙方（姓名）：
        r"甲乙双方.*?协商一致",     # 甲乙双方...协商一致
        r"根据.*?[《〈].*?[》〉].*?规定",  # 根据《法律》规定
        r"本合同.*?自.*?起生效",    # 本合同自...起生效
        r"签订本.*?合同",           # 签订本劳动合同
        r"经.*?友好协商",           # 经友好协商
        r"双方.*?本着.*?原则",      # 双方本着...原则
    ]

    @classmethod
    def is_opening_clause(cls, clause_text: str) -> bool:
        """
        检测是否为开场白条款（合同开头的程序性内容）。
        开场白条款通常不包含实质性风险。
        """
        # 检查是否匹配开场白模式
        opening_matches = 0
        for pattern in cls.OPENING_PATTERNS:
            if re.search(pattern, clause_text):
                opening_matches += 1
        
        # 如果匹配 2 个以上开场白特征，判定为开场白
        if opening_matches >= 2:
            return True
        
        # 额外检查：如果同时包含"甲方"+"乙方"+"签订/协商"，也判定为开场白
        if ("甲方" in clause_text and "乙方" in clause_text and 
            any(kw in clause_text for kw in ["签订", "协商一致", "订立", "本着"])):
            return True
        
        return False

    @classmethod
    def classify(cls, clause_text: str) -> Set[str]:
        categories = set()
        
        # 先检查是否为开场白
        if cls.is_opening_clause(clause_text):
            categories.add("opening")
            return categories  # 开场白直接返回，不做其他分类
        
        for cat, patterns in cls.CATEGORIES.items():
            for pattern in patterns:
                if re.search(pattern, clause_text):
                    categories.add(cat)
                    break
        
        if not categories:
            categories.add("other")
            
        return categories

class RiskFilter:
    """
    风险点过滤器：根据预处理结果过滤不适用的风险点。
    """
    
    @staticmethod
    def count_verbs(text: str) -> int:
        """统计动词数量 (粗略)"""
        words = pseg.cut(text)
        count = 0
        for w, flag in words:
            if flag.startswith('v'):
                count += 1
        return count
    
    @staticmethod
    def analyze_salary_structure(clause_text: str) -> dict:
        """
        分析薪资结构是否失衡。
        检测"高绩效低底薪"结构风险。
        
        Returns:
            dict: {
                'has_salary_info': bool,
                'base_salary': float or None,
                'performance_salary': float or None,
                'total_salary': float or None,
                'base_ratio': float or None (底薪占比),
                'is_imbalanced': bool (是否失衡),
                'warning_message': str or None
            }
        """
        import re
        
        result = {
            'has_salary_info': False,
            'base_salary': None,
            'performance_salary': None,
            'total_salary': None,
            'base_ratio': None,
            'is_imbalanced': False,
            'warning_message': None
        }
        
        # 匹配金额模式（支持多种格式）
        # 基本工资/底薪/基础工资
        base_patterns = [
            r'基本工资[为是]?\s*[：:]?\s*([\d,，]+(?:\.\d+)?)\s*元',
            r'底薪[为是]?\s*[：:]?\s*([\d,，]+(?:\.\d+)?)\s*元',
            r'基础工资[为是]?\s*[：:]?\s*([\d,，]+(?:\.\d+)?)\s*元',
            r'固定工资[为是]?\s*[：:]?\s*([\d,，]+(?:\.\d+)?)\s*元',
        ]
        
        # 绩效工资/浮动工资/提成
        performance_patterns = [
            r'绩效工资[为是]?\s*[：:]?\s*([\d,，]+(?:\.\d+)?)\s*元',
            r'浮动工资[为是]?\s*[：:]?\s*([\d,，]+(?:\.\d+)?)\s*元',
            r'绩效[为是]?\s*[：:]?\s*([\d,，]+(?:\.\d+)?)\s*元',
            r'提成[为是]?\s*[：:]?\s*([\d,，]+(?:\.\d+)?)\s*元',
        ]
        
        # 总工资/月薪
        total_patterns = [
            r'月工资[标准为是]*\s*[：:]?\s*([\d,，]+(?:\.\d+)?)\s*元',
            r'月薪[为是]?\s*[：:]?\s*([\d,，]+(?:\.\d+)?)\s*元',
            r'工资标准[为是]?\s*[：:]?\s*([\d,，]+(?:\.\d+)?)\s*元',
            r'税前[为是]?\s*[：:]?\s*([\d,，]+(?:\.\d+)?)\s*元',
        ]
        
        def extract_amount(patterns, text):
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    amount_str = match.group(1).replace(',', '').replace('，', '')
                    try:
                        return float(amount_str)
                    except ValueError:
                        pass
            return None
        
        base = extract_amount(base_patterns, clause_text)
        performance = extract_amount(performance_patterns, clause_text)
        total = extract_amount(total_patterns, clause_text)
        
        # 如果有基本工资和绩效工资，但没有总工资，计算总工资
        if base and performance and not total:
            total = base + performance
        
        # 如果有总工资和其中一项，推算另一项
        if total and base and not performance:
            performance = total - base
        if total and performance and not base:
            base = total - performance
        
        # 如果成功提取到薪资信息
        if base and total and total > 0:
            result['has_salary_info'] = True
            result['base_salary'] = base
            result['performance_salary'] = performance
            result['total_salary'] = total
            result['base_ratio'] = base / total
            
            # 判断是否失衡：底薪占比低于 40%
            if result['base_ratio'] < 0.4:
                result['is_imbalanced'] = True
                result['warning_message'] = (
                    f"⚠️ 薪资结构失衡警告：底薪 {base:.0f} 元仅占总薪资 {total:.0f} 元的 "
                    f"{result['base_ratio']*100:.0f}%（低于 40%）。"
                    f"这种'高绩效低底薪'结构可能导致：\n"
                    f"  1. 加班费计算基数偏低\n"
                    f"  2. 病假工资/经济补偿金基数缩水\n"
                    f"  3. 绩效工资被随意克扣的风险\n"
                    f"建议：确认该结构是否合理，或要求提高底薪占比。"
                )
        
        return result

    @classmethod
    def filter_rules(cls, clause_text: str, categories: Set[str], rules: List[Dict]) -> List[Dict]:
        """
        过滤逻辑：
        0. 开场白过滤：opening 类别直接返回空规则
        1. 结构过滤：信息/定义类条款若动词少，直接忽略。
        2. 适用性过滤：applicable_to
        3. 关键词过滤：required_keywords
        """
        
        # 0. 开场白过滤（合同开头的程序性内容，无风险）
        if "opening" in categories:
            return []  # 直接返回空，跳过所有规则匹配
        
        # 1. 结构过滤 (动词检测)
        # 如果仅包含 info 或 definition，且动词很少，视为低风险
        if categories.issubset({"info", "definition", "other"}) and not categories.intersection({"rights_obligations", "liability", "dispute", "termination", "change"}):
            verb_count = cls.count_verbs(clause_text)
            if verb_count < 1:
                return [] # 直接判定为无风险，不召回任何规则

        allowed_rules = []
        
        for rule in rules:
            # 2. 适用性过滤 (applicable_to)
            applicable_to = rule.get("applicable_to")
            if applicable_to:
                # 如果规则指定了适用类别，则条款类别必须在其中
                # 取交集：只要条款的一个类别在 applicable_to 中即可
                if not categories.intersection(set(applicable_to)):
                    continue
            
            # 3. must-have 关键词过滤 (required_keywords)
            required_keywords = rule.get("required_keywords")
            if required_keywords:
                # 必须包含至少一个 required_keyword
                has_keyword = False
                for kw in required_keywords:
                    if kw in clause_text:
                        has_keyword = True
                        break
                if not has_keyword:
                    continue
                    
            allowed_rules.append(rule)
            
        return allowed_rules

def preprocess_clause(clause_text: str, risk_library: List[Dict]) -> List[Dict]:
    """
    预处理总控函数。
    
    Args:
        clause_text: 条款文本
        risk_library: 完整风险规则库
        
    Returns:
        allowed_rules: 允许参与后续检索的风险点列表
    """
    # 1. 分类
    categories = ClauseClassifier.classify(clause_text)
    
    # 2. 过滤
    allowed_rules = RiskFilter.filter_rules(clause_text, categories, risk_library)
    
    return allowed_rules

# --- 示例风险库结构 ---
EXAMPLE_RISK_LIBRARY = [
    {
        "risk_id": "TEST_001",
        "risk_name": "单方变更权",
        "applicable_to": ["change", "rights_obligations"], # 仅适用于变更类或权利义务类
        "required_keywords": ["变更", "调整"], # 必须包含这些词之一
        "analysis_logic": "..."
    },
    {
        "risk_id": "TEST_002",
        "risk_name": "管辖陷阱",
        "applicable_to": ["dispute"], # 仅适用于争议解决类
        "required_keywords": ["法院", "仲裁"],
        "analysis_logic": "..."
    },
    {
        "risk_id": "TEST_003",
        "risk_name": "通用风险",
        "applicable_to": [], # 适用于所有 (空列表或不填)
        "analysis_logic": "..."
    }
]

if __name__ == "__main__":
    # 简单测试
    clause = "甲方有权单方变更本协议内容。"
    print(f"Clause: {clause}")
    cats = ClauseClassifier.classify(clause)
    print(f"Categories: {cats}")
    
    matches = preprocess_clause(clause, EXAMPLE_RISK_LIBRARY)
    print(f"Matched Rules: {[r['risk_name'] for r in matches]}")
    
    clause2 = "联系地址：北京市海淀区。"
    print(f"\nClause: {clause2}")
    cats2 = ClauseClassifier.classify(clause2)
    print(f"Categories: {cats2}")
    matches2 = preprocess_clause(clause2, EXAMPLE_RISK_LIBRARY)
    print(f"Matched Rules: {[r['risk_name'] for r in matches2]}")
