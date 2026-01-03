#!/usr/bin/env python
"""
规则库质量优化脚本
- 删除重复规则
- 扩充关键词
- 统一法条格式
"""

import json
import re
from pathlib import Path

RULES_PATH = Path("src/core/risk_rules.json")

def load_rules():
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_rules(rules):
    with open(RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=4)
    print(f"已保存 {len(rules)} 条规则")

# ============================================================================
# 1. 删除重复规则
# ============================================================================
DUPLICATE_IDS = [
    "LABOR_020",  # 与 LABOR_008 重复（末位淘汰）
    "LABOR_021",  # 与 LABOR_007 重复（加班费）
]

def remove_duplicates(rules):
    original_count = len(rules)
    rules = [r for r in rules if r.get("risk_id") not in DUPLICATE_IDS]
    removed = original_count - len(rules)
    print(f"删除 {removed} 条重复规则: {DUPLICATE_IDS}")
    return rules

# ============================================================================
# 2. 关键词扩充映射
# ============================================================================
KEYWORD_EXPANSIONS = {
    # --- 通用条款 ---
    "GENERAL_001": ["异地法院", "外地仲裁", "甲方所在地仲裁", "指定管辖"],
    "GENERAL_002": ["解释权归", "以...解释为准", "本公司解释", "甲方有权解释"],
    "GENERAL_003": ["到期自动续", "不通知视为续约", "自动扣款续期"],
    "GENERAL_004": ["单方调整", "有权修改", "保留变更权", "随时调整条款"],
    "GENERAL_005": ["责任上限", "损失不赔", "概不赔偿", "限于直接损失"],
    "GENERAL_006": ["政策调整", "市场变化", "第三方原因", "自然因素"],
    "GENERAL_007": ["向第三方披露", "共享个人信息", "用于商业推广", "授权使用数据"],
    "GENERAL_008": ["邮寄即视为送达", "公告视为通知", "电子邮件送达"],
    "GENERAL_009": ["单方解除", "无条件终止", "即时解约", "有权终止"],
    
    # --- 劳动合同 ---
    "LABOR_001": ["试用期超过", "延长试用", "二次试用", "单独试用期合同"],
    "LABOR_002": ["试用期薪资", "转正前工资", "试用工资低于"],
    "LABOR_003": ["不上社保", "工资含社保", "补贴代替社保"],
    "LABOR_004": ["竞业期限", "离职后不得从事", "同业禁止"],
    "LABOR_005": ["离职赔偿金", "提前走人罚款", "辞职扣钱"],
    "LABOR_006": ["调岗降薪", "异地调动", "岗位变动", "工作地点变更"],
    "LABOR_007": ["包含加班工资", "综合薪资", "不另付加班费"],
    "LABOR_008": ["业绩排名淘汰", "考核不达标辞退", "末尾辞退"],
    "LABOR_009": ["收取押金", "扣押学历证", "财物担保"],
    
    # --- 租赁合同 ---
    "LEASE_001": ["押金不予退还", "没收保证金", "押金转为赔偿"],
    "LEASE_002": ["修理费自理", "损坏自修", "设备维护由租客"],
    "LEASE_003": ["不可转租", "禁止分租", "不得出租他人"],
    "LEASE_005": ["买卖后搬离", "产权变更终止", "新房东有权收回"],
    "LEASE_006": ["租金上涨", "租金随市调整", "涨价权"],
    "LEASE_007": ["持有钥匙", "进入检查", "看房权"],
    "LEASE_008": ["退租扣押金", "提前走赔偿", "违约扣全款"],
    
    # --- 买卖合同 ---
    "SALES_001": ["定金罚没", "定金不予退还", "丧失定金"],
    "SALES_002": ["不退不换", "恕不退货", "一经售出"],
    "SALES_003": ["收货即验收", "签收视为验收合格", "当场检验"],
    "SALES_004": ["发货时间不定", "按库存发货", "物流时间不确定"],
    "SALES_005": ["物流风险买方", "运输损坏不赔", "快递问题自理"],
    "SALES_020": ["不适用无理由退货", "拆封不退", "促销品不退"],
}

def expand_keywords(rules):
    expanded_count = 0
    for rule in rules:
        risk_id = rule.get("risk_id")
        if risk_id in KEYWORD_EXPANSIONS:
            existing = set(rule.get("keywords", []))
            new_keywords = KEYWORD_EXPANSIONS[risk_id]
            for kw in new_keywords:
                if kw not in existing:
                    rule["keywords"].append(kw)
                    expanded_count += 1
    print(f"扩充 {expanded_count} 个关键词")
    return rules

# ============================================================================
# 3. 统一法条格式
# ============================================================================
def normalize_law_format(rules):
    """统一法条格式：确保 《法律名》第X条 格式"""
    fixed_count = 0
    for rule in rules:
        laws = rule.get("laws", "")
        if not laws:
            continue
        
        # 修复缺少"第"字的情况，如 《民法典》五百条 -> 《民法典》第五百条
        # 匹配 《XX》后紧跟中文数字但没有"第"字的情况
        pattern = r"《([^》]+)》([一二三四五六七八九十百千零\d]+[条款])"
        def add_di(match):
            law_name = match.group(1)
            article = match.group(2)
            if not article.startswith("第"):
                return f"《{law_name}》第{article}"
            return match.group(0)
        
        new_laws = re.sub(pattern, add_di, laws)
        if new_laws != laws:
            rule["laws"] = new_laws
            fixed_count += 1
    
    print(f"修正 {fixed_count} 条法条格式")
    return rules

# ============================================================================
# 主流程
# ============================================================================
def main():
    print("=" * 50)
    print("规则库质量优化")
    print("=" * 50)
    
    rules = load_rules()
    print(f"原始规则数: {len(rules)}")
    
    # 1. 删除重复
    rules = remove_duplicates(rules)
    
    # 2. 扩充关键词
    rules = expand_keywords(rules)
    
    # 3. 统一法条格式
    rules = normalize_law_format(rules)
    
    # 保存
    save_rules(rules)
    print("=" * 50)
    print("优化完成！")

if __name__ == "__main__":
    main()
