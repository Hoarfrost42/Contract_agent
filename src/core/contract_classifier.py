import re
from typing import Dict, List

class ContractClassifier:
    """
    简易合同类型分类器
    根据全文内容判断合同类型：劳动/租赁/买卖/借款/通用
    """
    
    # 类型映射定义
    # Key: 内部标识符 (对应 rules.json 中的 contract_type 值或其映射)
    # Value: 关键词列表
    TYPE_KEYWORDS = {
        "劳动合同": [
            r"劳动合同", r"聘用合同", r"雇佣", r"用人单位", r"劳动者", 
            r"试用期", r"工作地点", r"社会保险", r"公积金", r"竞业限制",r"劳动"
        ],
        "租赁合同": [
            r"租赁合同", r"房屋租赁", r"承租方", r"出租方", r"租赁期限",
            r"租金支付", r"押金", r"物业费", r"转租",r"租赁"
        ],
        "买卖合同": [
            r"买卖合同", r"采购合同", r"供货合同", r"销售合同",
            r"买受人", r"出卖人", r"供方", r"需方", r"购销",r"标的物","买卖"
        ],
        "借款合同": [
            r"借款合同", r"贷款合同", r"借贷", r"出借人", r"借款人",
            r"利率", r"利息", r"还款", r"抵押"
        ],
        # "服务合同": [r"技术服务", r"咨询服务", r"委托开发"] # 暂未启用
    }
    
    # 规则库中的 contract_type 如果包含 "/"，如 "通用/租赁/买卖"，需要特殊处理
    # 这里我们只返回单一的主类型
    
    @classmethod
    def classify(cls, text: str) -> str:
        """
        判断合同类型
        
        Args:
            text: 合同全文或前 2000 字
            
        Returns:
            str: "劳动合同", "租赁合同", "买卖合同", "借款合同" 或 "通用"
        """
        # 截取前 3000 字进行判断（通常合同类型在开头或标题已明确）
        sample_text = text[:3000]
        
        scores: Dict[str, int] = {k: 0 for k in cls.TYPE_KEYWORDS}
        
        for type_name, keywords in cls.TYPE_KEYWORDS.items():
            for kw in keywords:
                # 简单计数，关键词出现一次 +1
                # 使用 finditer 计算出现次数
                count = len(list(re.finditer(kw, sample_text)))
                if "合同" in kw and count > 0:
                    # 标题类关键词权重更高
                    scores[type_name] += count * 5
                else:
                    scores[type_name] += count
        
        # 找出得分最高的类型
        if not scores:
            return "通用"
            
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        
        # 阈值判断：如果最高分太低，归为通用
        if max_score < 3:
            return "通用"
            
        return best_type
