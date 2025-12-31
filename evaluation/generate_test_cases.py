#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 risk_rules.json 生成测试用例

基于专家规则库生成正例、负例和边界用例，用于评测合同风险识别系统。

使用方法:
    python generate_test_cases.py --output generated_test_cases.jsonl
    python generate_test_cases.py --use-llm --output generated_test_cases.jsonl  # 使用 LLM 增强生成
"""

import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 合同条款模板 - 用于基于规则关键词生成测试用例
CLAUSE_TEMPLATES = {
    "positive": [
        # 正例模板 - 关键词直接嵌入
        "本合同约定，{keyword}。双方应遵守此条款。",
        "甲方有权{keyword}，乙方不得提出异议。",
        "双方同意，{keyword}，此条款为合同生效条件。",
        "关于争议解决，{keyword}。",
        "如发生纠纷，{keyword}。",
        "乙方承诺{keyword}，违反者需承担违约责任。",
        "{keyword}，该规定自合同签订之日起生效。",
        "除非另有约定，{keyword}。",
    ],
    "negative": [
        # 负例模板 - 不包含风险关键词的正常条款
        "本合同的履行地点为双方协商确定的地点。",
        "双方应本着诚实信用原则履行本合同。",
        "本合同一式两份，双方各执一份，具有同等法律效力。",
        "如需变更合同内容，须经双方协商一致并签订书面补充协议。",
        "合同期限届满前30日，双方可协商续签事宜。",
        "甲方应按约定时间交付货物，乙方应按约定时间支付货款。",
        "本合同的解释应遵循中华人民共和国法律法规。",
        "任何一方违约，应向守约方赔偿因此造成的直接损失。",
    ],
    "boundary": [
        # 边界用例模板 - 部分匹配或模糊措辞
        "甲方{partial_keyword}，但应提前通知乙方。",
        "关于{partial_keyword}的事宜，双方另行协商。",
        "若出现{partial_keyword}情形，可协商处理。",
        "在特殊情况下，{partial_keyword}（需双方同意）。",
    ]
}

# 合同类型映射
CONTRACT_TYPES = {
    "通用条款": "general",
    "劳动合同": "labor",
    "劳务协议": "labor",
    "培训协议": "labor",
    "房屋租赁": "rental",
    "房屋买卖": "property",
    "借款合同": "loan",
    "服务合同": "service",
    "购销合同": "sales",
}


def load_risk_rules(rules_path: Path) -> list:
    """加载风险规则库"""
    with open(rules_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_positive_case(rule: dict, template_idx: int = None) -> dict:
    """
    生成正例 - 包含风险关键词的条款
    """
    keywords = rule.get("keywords", [])
    if not keywords:
        return None
    
    # 随机选择一个关键词
    keyword = random.choice(keywords)
    
    # 选择模板
    templates = CLAUSE_TEMPLATES["positive"]
    if template_idx is None:
        template_idx = random.randint(0, len(templates) - 1)
    template = templates[template_idx % len(templates)]
    
    # 生成条款文本
    clause_text = template.format(keyword=keyword)
    
    return {
        "id": f"{rule['risk_id']}_pos_{template_idx or 0}",
        "contract_text": clause_text,
        "expected_risks": [{
            "risk_id": rule["risk_id"],
            "risk_name": rule["risk_name"],
            "risk_type": rule.get("risk_type", "unknown"),
            "contract_type": rule.get("contract_type", "通用条款"),
            "triggered_keyword": keyword,
            "laws": rule.get("laws", ""),
        }],
        "case_type": "positive",
        "source": "rule_generated",
        "rule_id": rule["risk_id"],
        "generated_at": datetime.now().isoformat(),
    }


def generate_negative_case(rule: dict, idx: int = 0) -> dict:
    """
    生成负例 - 不包含该规则风险的正常条款
    """
    templates = CLAUSE_TEMPLATES["negative"]
    clause_text = templates[idx % len(templates)]
    
    return {
        "id": f"{rule['risk_id']}_neg_{idx}",
        "contract_text": clause_text,
        "expected_risks": [],  # 负例无风险
        "case_type": "negative",
        "source": "rule_generated",
        "rule_id": rule["risk_id"],
        "target_risk_id": rule["risk_id"],  # 标注这是针对哪条规则的负例
        "generated_at": datetime.now().isoformat(),
    }


def generate_boundary_case(rule: dict, idx: int = 0) -> dict:
    """
    生成边界用例 - 部分匹配或模糊措辞
    """
    keywords = rule.get("keywords", [])
    if not keywords:
        return None
    
    # 选择一个关键词并截取部分
    keyword = random.choice(keywords)
    # 取关键词的前半部分作为部分匹配
    partial_keyword = keyword[:len(keyword)//2+2] if len(keyword) > 4 else keyword
    
    templates = CLAUSE_TEMPLATES["boundary"]
    template = templates[idx % len(templates)]
    clause_text = template.format(partial_keyword=partial_keyword)
    
    return {
        "id": f"{rule['risk_id']}_boundary_{idx}",
        "contract_text": clause_text,
        "expected_risks": [],  # 边界用例可能触发也可能不触发
        "case_type": "boundary",
        "source": "rule_generated",
        "rule_id": rule["risk_id"],
        "requires_review": True,  # 边界用例需要人工审核
        "generated_at": datetime.now().isoformat(),
    }


def generate_all_test_cases(
    rules: list,
    positive_per_rule: int = 2,
    negative_per_rule: int = 1,
    boundary_per_rule: int = 1,
) -> list:
    """
    从规则库生成所有测试用例
    
    Args:
        rules: 风险规则列表
        positive_per_rule: 每条规则生成的正例数量
        negative_per_rule: 每条规则生成的负例数量  
        boundary_per_rule: 每条规则生成的边界用例数量
    
    Returns:
        测试用例列表
    """
    test_cases = []
    
    for rule in rules:
        # 生成正例
        for i in range(positive_per_rule):
            case = generate_positive_case(rule, i)
            if case:
                test_cases.append(case)
        
        # 生成负例
        for i in range(negative_per_rule):
            case = generate_negative_case(rule, i)
            if case:
                test_cases.append(case)
        
        # 生成边界用例
        for i in range(boundary_per_rule):
            case = generate_boundary_case(rule, i)
            if case:
                test_cases.append(case)
    
    return test_cases


def enhance_with_llm(cases: list, model: str = "qwen2.5:14b") -> list:
    """
    使用 LLM 增强测试用例（可选功能）
    
    通过 LLM 重写条款使其更加自然、多样化
    """
    import requests
    
    def call_ollama(prompt: str) -> str:
        """直接调用 Ollama API"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=120
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                print(f"  Ollama API 错误: {response.status_code}")
                return ""
        except Exception as e:
            print(f"  Ollama 调用失败: {e}")
            return ""
    
    enhanced_cases = []
    
    prompt_template = """你是一个合同条款改写专家。请将以下合同条款改写为更自然、更符合真实合同风格的版本，保留关键信息不变。

原始条款：{clause}

要求：
1. 保持原有的法律含义和风险点
2. 使用更正式的合同用语
3. 控制在100字以内

改写后的条款："""
    
    print(f"使用 LLM ({model}) 增强 {len(cases)} 条测试用例...")
    
    for i, case in enumerate(cases):
        if case["case_type"] == "positive":  # 只增强正例
            try:
                prompt = prompt_template.format(clause=case["contract_text"])
                enhanced_text = call_ollama(prompt).strip()
                if enhanced_text and len(enhanced_text) > 10:
                    case["contract_text_original"] = case["contract_text"]
                    case["contract_text"] = enhanced_text
                    case["llm_enhanced"] = True
            except Exception as e:
                print(f"  警告: 第 {i+1} 条增强失败: {e}")
        
        enhanced_cases.append(case)
        
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i+1}/{len(cases)} 条")
    
    return enhanced_cases


def save_test_cases(cases: list, output_path: Path):
    """保存测试用例为 JSONL 格式"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
    print(f"已保存 {len(cases)} 条测试用例到 {output_path}")


def print_statistics(cases: list):
    """打印测试用例统计信息"""
    total = len(cases)
    positive = sum(1 for c in cases if c["case_type"] == "positive")
    negative = sum(1 for c in cases if c["case_type"] == "negative")
    boundary = sum(1 for c in cases if c["case_type"] == "boundary")
    
    # 按合同类型统计
    by_contract_type = {}
    for case in cases:
        if case["expected_risks"]:
            ct = case["expected_risks"][0].get("contract_type", "未知")
        else:
            ct = "负例/边界"
        by_contract_type[ct] = by_contract_type.get(ct, 0) + 1
    
    print("\n" + "="*50)
    print("测试用例生成统计")
    print("="*50)
    print(f"总数: {total}")
    print(f"  - 正例: {positive}")
    print(f"  - 负例: {negative}")
    print(f"  - 边界用例: {boundary}")
    print("\n按合同类型分布:")
    for ct, count in sorted(by_contract_type.items(), key=lambda x: -x[1]):
        print(f"  - {ct}: {count}")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="从 risk_rules.json 生成测试用例")
    parser.add_argument(
        "--rules", 
        type=str, 
        default=str(PROJECT_ROOT / "src" / "core" / "risk_rules.json"),
        help="风险规则文件路径"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="generated_test_cases.jsonl",
        help="输出文件名"
    )
    parser.add_argument(
        "--positive", 
        type=int, 
        default=2,
        help="每条规则生成的正例数量"
    )
    parser.add_argument(
        "--negative", 
        type=int, 
        default=1,
        help="每条规则生成的负例数量"
    )
    parser.add_argument(
        "--boundary", 
        type=int, 
        default=1,
        help="每条规则生成的边界用例数量"
    )
    parser.add_argument(
        "--use-llm", 
        action="store_true",
        help="使用 LLM 增强生成的测试用例"
    )
    parser.add_argument(
        "--llm-model", 
        type=str, 
        default="qwen2.5:14b",
        help="LLM 模型名称"
    )
    
    args = parser.parse_args()
    
    # 加载规则
    rules_path = Path(args.rules)
    if not rules_path.exists():
        print(f"错误: 规则文件不存在: {rules_path}")
        return 1
    
    print(f"加载规则文件: {rules_path}")
    rules = load_risk_rules(rules_path)
    print(f"共载入 {len(rules)} 条风险规则")
    
    # 生成测试用例
    print(f"\n生成测试用例 (正例:{args.positive}, 负例:{args.negative}, 边界:{args.boundary})...")
    test_cases = generate_all_test_cases(
        rules,
        positive_per_rule=args.positive,
        negative_per_rule=args.negative,
        boundary_per_rule=args.boundary,
    )
    
    # LLM 增强 (可选)
    if args.use_llm:
        test_cases = enhance_with_llm(test_cases, model=args.llm_model)
    
    # 统计信息
    print_statistics(test_cases)
    
    # 保存结果
    output_path = Path(__file__).parent / args.output
    save_test_cases(test_cases, output_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
