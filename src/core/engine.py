
import asyncio
import re
import logging
from typing import List, Dict, Any

from src.core.llm import LLMClient
from src.core.types import ClauseAnalysis
from src.core.rule_engine import RuleEngine
from src.core.contract_classifier import ContractClassifier
from src.utils.parser import split_contract
from src.utils.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

class ContractAnalyzer:
    def __init__(self):
        # 初始化进度追踪器
        self.tracker = ProgressTracker()
        # 初始化规则引擎
        self.rule_engine = RuleEngine()

    async def analyze(self, job_id: str, text: str, llm_source: str = "local", deep_reflection: bool = False):
        """
        主分析工作流 (基于规则+大模型重构)。
        
        流程：
        1. 文本切分：将合同文本切分为独立条款。
        2. 并行处理：
           a. 规则匹配：使用规则引擎匹配风险点。
           b. LLM分析：结合规则信息进行深度分析。
           c. (可选) 自反思：对高风险结果进行二次审查。
        3. 报告生成：汇总分析结果，生成摘要和详细报告。
        
        Args:
            deep_reflection: 是否启用深度反思模式（对高风险结果进行二次审查）
        """
        mode_desc = "深度反思模式" if deep_reflection else "标准模式"
        self.tracker.add_log(job_id, f"开始分析合同 (模型源: {llm_source}, 模式: {mode_desc})...")
        
        # 初始化 LLM 客户端
        llm_client = LLMClient(source=llm_source)
        
        # 0. 合同类型推断
        contract_type = ContractClassifier.classify(text)
        self.tracker.add_log(job_id, f"🔍 识别合同类型为: {contract_type} (将根据此类型过滤无关风险规则)")
        
        # 1. 使用正则切分合同条款
        clauses_text = split_contract(text)
        if not clauses_text:
            self.tracker.set_result(job_id, {"error": "无法解析合同文本"})
            return

        self.tracker.add_log(job_id, f"共识别出 {len(clauses_text)} 个条款，开始并行分析...")

        all_clauses: List[ClauseAnalysis] = []
        all_results_dicts: List[Dict[str, Any]] = []

        # 加载并发限制配置
        from src.utils.config_loader import load_config
        config = load_config()
        max_concurrency = config.get("system_config", {}).get("max_concurrency", 5)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_clause(i: int, clause_text: str):
            """处理单个条款的内部异步函数"""
            async with semaphore:
                # 2. 规则匹配 (Rule Matching) - 使用统一检索器
                from src.core.reference_retriever import retrieve_reference
                # 传入 contract_type 进行精准检索
                result = retrieve_reference(clause_text, contract_type=contract_type)
                
                reference_info = result.reference_info
                law_contents = result.law_contents
                risk_ids = result.risk_ids
                scores = result.scores
                
                # 兼容：取第一个匹配结果用于后续处理
                law_content = law_contents[0] if law_contents else None
                risk_id = risk_ids[0] if risk_ids else None
                confidence = scores[0] if scores else 0.0
                match_source = result.match_source
                
                # ========== 打印规则匹配结果（人工审查）==========
                clause_preview = clause_text[:60].replace('\n', ' ') + "..." if len(clause_text) > 60 else clause_text.replace('\n', ' ')
                print(f"\n{'='*60}")
                print(f"📋 条款 {i+1}: {clause_preview}")
                
                if risk_ids:
                    print(f"   ✅ 匹配到 {len(risk_ids)} 个候选规则 (reranked={result.reranked})")
                    for j, (rid, score) in enumerate(zip(risk_ids, scores)):
                        rule = next((r for r in self.rule_engine.rules if r.get('risk_id') == rid), None)
                        rule_name = rule.get('risk_name', '未知') if rule else '未知'
                        print(f"      [{j+1}] {rid}: {rule_name} (置信度: {score:.2f})")
                else:
                    print(f"   ❌ 无匹配规则")
                print(f"{'='*60}")
                
                # 3. LLM 分析 (LLM Analysis)
                # 调用大模型进行单条款分析，传入条款文本和参考信息
                clause_analysis = await asyncio.to_thread(
                    llm_client.analyze_clause, 
                    clause_text, 
                    reference_info
                )
                
                # 注入检索到的法律原文和风险ID
                if clause_analysis:
                    if law_content:
                        clause_analysis.law_content = law_content
                    if risk_id:
                        clause_analysis.risk_id = risk_id
                    
                    # --- 关键词兜底触发：添加人工复核提示 ---
                    if match_source.startswith("keyword_fallback:"):
                        keyword = match_source.split(":", 1)[1]
                        clause_analysis.suggestion = f"⚠️ 检测到高危关键词「{keyword}」，建议人工复核。" + (clause_analysis.suggestion or "")
                        # 不强制判定为高风险，保留 LLM 的判断，但添加提示
                    
                    # --- 薪资结构失衡检测 ---
                    # 检测"高绩效低底薪"结构风险
                    from src.core.preprocessor import RiskFilter
                    salary_analysis = RiskFilter.analyze_salary_structure(clause_text)
                    if salary_analysis['is_imbalanced']:
                        print(f"\n💰 检测到薪资结构失衡：底薪占比 {salary_analysis['base_ratio']*100:.0f}%")
                        # 如果当前是低风险或无风险，升级为中风险
                        if clause_analysis.risk_level == "低":
                            clause_analysis.risk_level = "中"
                            clause_analysis.risk_reason = "薪资结构失衡（底薪占比过低）"
                        # 添加详细警告到建议中
                        clause_analysis.suggestion = (
                            salary_analysis['warning_message'] + "\n\n" + 
                            (clause_analysis.suggestion or "")
                        )
                    
                    # --- 深度反思模式（可选）---
                    # 对高风险和中风险结果进行二次审查，降低幻觉风险
                    # 传入：条款原文 + LLM分析结果 + 原始规则信息（包含后果分析）
                    if deep_reflection and clause_analysis.risk_level in ["高", "中"]:
                        print(f"\n{'🔄'*20}")
                        print(f"🧠 触发深度反思 - 条款 {i+1}")
                        print(f"   原判定: {clause_analysis.risk_level}风险 | 原因: {clause_analysis.risk_reason[:50]}...")
                        print(f"   正在进行二次审查...")
                        
                        conclusion, reason = await asyncio.to_thread(
                            llm_client.self_reflect, 
                            clause_analysis,
                            reference_info  # 传入原始规则信息
                        )
                        
                        print(f"   ✅ 反思结论: 【{conclusion}】")
                        print(f"   📝 理由: {reason}")
                        print(f"{'🔄'*20}\n")
                        
                        # 处理调级（支持双向调整）
                        if conclusion in ["调级", "降级"]:
                            # 尝试从理由中解析目标等级
                            original_level = clause_analysis.risk_level
                            if "高" in reason and original_level != "高":
                                clause_analysis.risk_level = "高"
                                clause_analysis.suggestion = f"[自反思调级→高] {reason}。{clause_analysis.suggestion or ''}"
                            elif "中" in reason and original_level != "中":
                                clause_analysis.risk_level = "中"
                                clause_analysis.suggestion = f"[自反思调级→中] {reason}。{clause_analysis.suggestion or ''}"
                            elif "低" in reason or original_level == "高":
                                # 默认降级逻辑：高→中 或 中→低
                                if original_level == "高":
                                    clause_analysis.risk_level = "中"
                                    clause_analysis.suggestion = f"[自反思调级→中] {reason}。{clause_analysis.suggestion or ''}"
                                else:
                                    clause_analysis.risk_level = "低"
                                    clause_analysis.suggestion = f"[自反思调级→低] {reason}。{clause_analysis.suggestion or ''}"
                        elif conclusion == "存疑":
                            clause_analysis.suggestion = f"⚠️ [自反思存疑] {reason}，建议人工复核。{clause_analysis.suggestion or ''}"
                        
                    # --- 低风险标准化处理 ---
                    if clause_analysis.risk_level == "低":
                        clause_analysis.risk_reason = "条款内容未涉及典型风险点，未发现明显法律风险。"
                        clause_analysis.deep_analysis = "条款主要为信息性或程序性表述，不包含单方变更、显失公平或违反强制性规定的内容，因此风险较低。"
                        clause_analysis.suggestion = "无须修改，条款表述已符合法律要求。"
                        # 清空法律引用，确保前端不显示
                        clause_analysis.law_reference = ""
                        clause_analysis.law_content = ""
                
                return clause_analysis

        # 并行执行所有条款的分析任务
        tasks = [process_clause(i, c) for i, c in enumerate(clauses_text)]
        results = await asyncio.gather(*tasks)

        # 过滤掉无效结果 (无风险或分析失败的)
        valid_results = [r for r in results if r is not None]
        
        self.tracker.add_log(job_id, f"分析完成，发现 {len(valid_results)} 个风险点。")

        for r in valid_results:
            all_clauses.append(r)
            all_results_dicts.append(r.dict())

        # 4. 生成统一报告
        self.tracker.add_log(job_id, "正在生成汇总报告...")
        final_report = await self._generate_final_report(all_clauses, llm_client)

        # 计算风险评分 (高风险=10分，低风险=2分，最高100分)
        high_risk_count = sum(1 for r in all_clauses if r.risk_level == "高")
        low_risk_count = sum(1 for r in all_clauses if r.risk_level == "低")
        risk_score = min(100, high_risk_count * 10 + low_risk_count * 2)

        result = {
            "report": final_report,
            "results": all_results_dicts,
            "risks": all_results_dicts,  # 兼容 state.py 的 key
            "summary": final_report,      # 兼容 state.py 的 key
            "risk_score": risk_score,
        }

        self.tracker.set_result(job_id, result)
        
        # 释放 LLM 资源
        self.tracker.add_log(job_id, "正在释放模型资源...")
        llm_client.unload_model()
        
        # 返回结果供调用者使用
        return result

    def _generate_summary_text(self, clauses: List[ClauseAnalysis]) -> str:
        """
        使用 Python 逻辑生成执行摘要，替代 LLM 生成。
        
        功能：
        1. 统计高/低风险数量。
        2. 提取前 5 个高风险条款的简述。
        3. 生成一段总结性文字。
        """
        high_risks = [c for c in clauses if "高" in c.risk_level]
        low_risks = [c for c in clauses if "低" in c.risk_level]
        
        total_risks = len(high_risks) + len(low_risks)
        
        if total_risks == 0:
            return "本次审查未发现显著风险条款，合同整体合规性良好。"
            
        summary = f"本次审查共扫描 {len(clauses)} 个条款，发现 {total_risks} 项风险点（高风险 {len(high_risks)} 项，低风险 {len(low_risks)} 项）。\n\n"
        
        if high_risks:
            summary += "### 核心风险预警 (Top 5)\n"
            for i, c in enumerate(high_risks[:5]):
                reason = c.risk_reason if c.risk_reason else "未说明具体原因"
                summary += f"{i+1}. **{reason}**\n"
            if len(high_risks) > 5:
                summary += f"...以及其他 {len(high_risks)-5} 项高风险条款。\n"
        
        summary += "\n建议重点关注上述高风险条款，并参考详细审查意见进行修改。"
        return summary

    async def _generate_final_report(self, clauses: List[ClauseAnalysis], llm_client: LLMClient) -> str:
        """
        生成最终的 Markdown 格式报告。
        
        包含：
        1. 执行摘要 (由 _generate_summary_text 生成)。
        2. 详细条款审查 (遍历所有风险条款并格式化)。
        """
        if not clauses:
            return "# 合同风险审查报告\n\n未发现显著风险条款。"

        # 1. 执行摘要 (Python 生成)
        summary = self._generate_summary_text(clauses)
        
        # 2. 详细分析
        details = ["# 合同风险审查报告", "", summary, "", "## 详细条款审查", ""]
        
        for idx, c in enumerate(clauses):
            # 移除 Emoji，使用纯文本标记
            risk_label = "【高风险】" if "高" in c.risk_level else "【低风险】"
            
            details.append(f"### {risk_label} 风险点 {idx+1}")
            details.append(f"**条款原文**：\n> {c.clause_text}")
            details.append(f"\n**风险等级**：{c.risk_level}")
            details.append(f"**风险简述**：{c.risk_reason}")
            
            if c.deep_analysis:
                details.append(f"\n**详细分析**：{c.deep_analysis}")
            
            # 只显示法条原文（已包含法律依据标题）
            if c.law_content:
                details.append(f"\n**法律依据**：\n> {c.law_content}")
                
            if c.suggestion:
                details.append(f"\n**修改建议**：{c.suggestion}")
                
            details.append("\n---")

        return "\n".join(details)

    def _deduplicate_clauses(self, clauses: List[ClauseAnalysis]) -> List[ClauseAnalysis]:
        # 在新流水线中已弃用，因为我们处理的是 split_contract 切分出的唯一条款
        return clauses
