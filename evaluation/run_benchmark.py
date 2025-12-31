import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.engine import ContractAnalyzer
from src.core.llm import LLMClient
from src.utils.parser import chunk_contract

JUDGE_PROMPT = """
你是一个公正的法律评估专家。请评估 AI 生成的风险分析理由是否准确。

### 评估输入
- **条款原文**: {clause}
- **标准答案 (Ground Truth)**: {ground_truth_keywords}
- **AI 生成理由**: {ai_reason}

### 评分标准
- **1分 (准确)**: AI 的理由包含了标准答案中的核心关键词或逻辑。
- **0分 (错误)**: AI 的理由完全偏离，或未识别出核心风险。

### 输出格式
仅输出一个数字：1 或 0
"""

async def evaluate_reasoning(llm, clause, ground_truth_keywords, ai_reason):
    if not ai_reason or not ground_truth_keywords:
        return 0
    
    prompt = JUDGE_PROMPT.format(
        clause=clause,
        ground_truth_keywords=", ".join(ground_truth_keywords),
        ai_reason=ai_reason
    )
    
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = getattr(response, "content", str(response)).strip()
        if "1" in content:
            return 1
        return 0
    except Exception as e:
        print(f"Judge error: {e}")
        return 0

async def evaluate_single_contract(analyzer, judge_llm, item: Dict[str, Any]) -> Dict[str, Any]:
    text = item.get("text", "")
    ground_truth = item.get("ground_truth", {})
    
    # Run Agent
    # We need to simulate the analyzer's process but just for one chunk/clause
    # Since ContractAnalyzer.analyze is designed for full documents and updates a tracker,
    # we might want to use the LLMClient directly for single clause evaluation to match the old graph behavior,
    # OR we can use the analyzer but capture the results differently.
    
    # The old graph.ainvoke(state) returned "scan_results".
    # The new analyzer.analyze returns nothing but updates the tracker.
    # However, for evaluation, we usually want to test the *scanning* capability primarily.
    
    # Let's use LLMClient directly to scan the text, similar to how engine.py does it.
    # This avoids the overhead of the full analyzer workflow (tracker, deduplication, etc.) which might be overkill for single-clause eval.
    
    llm_client = LLMClient(source="local") # Use local for eval by default
    
    try:
        # Analyze the text directly using analyze_clause
        # Note: analyze_clause returns a single ClauseAnalysis object or None
        from src.core.rule_engine import RuleEngine
        rule_engine = RuleEngine()
        
        # Get reference info from rule engine (simplified for eval)
        matched_rule, confidence, match_source = rule_engine.match_risk(text)
        reference_info = ""
        if matched_rule:
            reference_info = f"**{matched_rule.get('risk_name', '')}**\n{matched_rule.get('analysis_logic', '')}\n"
        elif match_source.startswith("keyword_fallback:"):
            keyword = match_source.split(":", 1)[1]
            reference_info = f"【关键词预警】检测到高危关键词：\"{keyword}\"，建议谨慎判断。\n"
        
        result = await asyncio.to_thread(llm_client.analyze_clause, text, reference_info)
        
        # Convert to list format to match old format
        scan_results = []
        if result:
            scan_results.append({
                "clause": result.clause_text,
                "risk": result.risk_level,
                "dimension": result.dimension,
                "reason": result.risk_reason,
            })
            
    except Exception as e:
        print(f"Agent error: {e}")
        scan_results = []
        
    # Evaluate
    # Assuming one clause per text in golden dataset
    if not scan_results:
        return {
            "id": item.get("id"),
            "correct_risk": False,
            "correct_dimension": False,
            "correct_reason": False,
            "prediction": {},
            "ground_truth": ground_truth
        }
        
    pred = scan_results[0] # Take the first one
    
    # 1. Risk Level Match
    # 当前系统使用"高/低"二元分类，Ground Truth 中的"中"视为"低"
    pred_risk = pred.get("risk", "未知")
    gt_risk = ground_truth.get("risk_level", "")
    
    # 将 Ground Truth 中的"中"映射为"低"（当前系统不支持中风险）
    if gt_risk == "中":
        gt_risk = "低"
    
    # 直接比较风险等级
    is_risk_correct = gt_risk in pred_risk or pred_risk == gt_risk
    
    # 2. Dimension Match
    pred_dim = str(pred.get("dimension", "0"))
    gt_dim = str(ground_truth.get("dimension_id", "0"))
    is_dim_correct = pred_dim == gt_dim
    
    # 3. Reasoning Match (LLM Judge)
    is_reason_correct = await evaluate_reasoning(
        judge_llm, 
        text, 
        ground_truth.get("reason_keywords", []), 
        pred.get("reason", "")
    )
    
    return {
        "id": item.get("id"),
        "correct_risk": is_risk_correct,
        "correct_dimension": is_dim_correct,
        "correct_reason": bool(is_reason_correct),
        "prediction": pred,
        "ground_truth": ground_truth
    }

async def run_benchmark(data_path: str, limit: int = None, log_callback=None):
    def log(msg):
        print(msg)
        if log_callback:
            log_callback(msg)

    log(f"Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        log(f"Error: Data file not found at {data_path}")
        return None

    # Load Data
    dataset = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
                
    if limit:
        dataset = dataset[:limit]
        
    log(f"Loaded {len(dataset)} samples.")
    
    # Init Judge
    # We can use LLMClient's text_llm for judging
    client = LLMClient(source="local")
    judge_llm = client.text_llm
    
    # Analyzer is not strictly needed as an object if we use LLMClient directly in evaluate_single_contract
    # but we can pass None or keep the signature
    analyzer = None 
    
    results = []
    
    for i, item in enumerate(dataset):
        log(f"Evaluating sample {i+1}/{len(dataset)}: {item.get('id')}...")
        res = await evaluate_single_contract(analyzer, judge_llm, item)
        results.append(res)
        
    # Statistics
    total = len(results)
    risk_acc = sum(1 for r in results if r["correct_risk"]) / total if total else 0
    dim_acc = sum(1 for r in results if r["correct_dimension"]) / total if total else 0
    reason_acc = sum(1 for r in results if r["correct_reason"]) / total if total else 0
    
    log("\n" + "="*30)
    log("📊 Evaluation Report")
    log("="*30)
    log(f"Total Samples: {total}")
    log(f"Risk Level Accuracy: {risk_acc:.2%}")
    log(f"Dimension ID Accuracy: {dim_acc:.2%}")
    log(f"Reasoning Quality Score: {reason_acc:.2%}")
    
    # Save Results with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("evaluation", exist_ok=True)
    result_file = f"evaluation/results_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"Results saved to {result_file}")
        
    # Export Bad Cases
    bad_cases = [r for r in results if not (r["correct_risk"] and r["correct_dimension"] and r["correct_reason"])]
    
    with open("evaluation/bad_cases.md", "w", encoding="utf-8") as f:
        f.write("# 🚨 Bad Cases Analysis\n\n")
        for case in bad_cases:
            f.write(f"### Case ID: {case['id']}\n")
            f.write(f"- **Clause (Pred)**: {case['prediction'].get('clause', 'N/A')}\n")
            f.write(f"- **Ground Truth**: Risk={case['ground_truth'].get('risk_level')}, Dim={case['ground_truth'].get('dimension_id')}, Keywords={case['ground_truth'].get('reason_keywords')}\n")
            f.write(f"- **Prediction**: Risk={case['prediction'].get('risk')}, Dim={case['prediction'].get('dimension')}\n")
            f.write(f"- **Reasoning**: {case['prediction'].get('reason')}\n")
            f.write(f"- **Errors**: ")
            errors = []
            if not case["correct_risk"]: errors.append("Risk Mismatch")
            if not case["correct_dimension"]: errors.append("Dimension Mismatch")
            if not case["correct_reason"]: errors.append("Reasoning Poor")
            f.write(", ".join(errors) + "\n\n")
            f.write("---\n")
            
    log(f"\nBad cases exported to evaluation/bad_cases.md ({len(bad_cases)} cases)")
    
    return {
        "metrics": {
            "total": total,
            "risk_acc": risk_acc,
            "dim_acc": dim_acc,
            "reason_acc": reason_acc
        },
        "results": results,
        "bad_cases": bad_cases
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="evaluation/golden_dataset.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(args.data, args.limit))
