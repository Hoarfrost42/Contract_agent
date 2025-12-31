import re
import textwrap
import streamlit as st
from typing import Dict, Any, List
from src.utils.prompt_manager import get_risk_dimensions

def render_saas_report(report_md: str, structured_data: List[Dict[str, Any]], back_callback=None):
    """
    交互式报告渲染 (Tabs + Expander 版)
    """
    
    # 0. 加载维度映射
    DIM_MAP = get_risk_dimensions()

    # 1. 辅助函数：Markdown 转 HTML
    def simple_md_to_html(text):
        if not text: return ""
        text = str(text)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text) # 加粗
        text = text.replace('\n', '<br>') # 换行
        return text

    # 2. 辅助函数：鲁棒的深度分析解析
    def parse_deep_analysis(text):
        data = {"violation": "", "consequence": "", "suggestion": ""}
        if not text: return data
        
        pattern = re.compile(r"【(违规点|违规|风险点|后果|风险后果|法律后果|建议|修改建议|优化建议)】")
        matches = list(pattern.finditer(text))
        
        if not matches:
            data["violation"] = text
            return data
            
        for i, match in enumerate(matches):
            tag = match.group(1)
            start = match.end()
            if i < len(matches) - 1:
                end = matches[i+1].start()
                raw_content = text[start:end]
                cleaned_content = re.sub(r"(\s*[\d\.\、\*]+\s*)$", "", raw_content)
            else:
                end = len(text)
                cleaned_content = text[start:end]
            
            content = cleaned_content.strip().lstrip("：: ")
            
            if tag in ["违规点", "违规", "风险点"]:
                data["violation"] = content
            elif tag in ["后果", "风险后果", "法律后果"]:
                data["consequence"] = content
            elif tag in ["建议", "修改建议", "优化建议"]:
                data["suggestion"] = content
                
        return data

    # 3. CSS 样式 (保持原有 Card 样式)
    custom_css = textwrap.dedent("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        .report-container { font-family: 'Inter', sans-serif; color: #1F2937; padding: 20px; max-width: 1200px; margin: 0 auto; }
        
        .dashboard-row { display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap;}
        
        .score-card { flex: 1; min-width: 200px; background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border: 1px solid #F3F4F6; text-align: center; display: flex; flex-direction: column; justify-content: center; }
        .score-circle { width: 80px; height: 80px; border-radius: 50%; background: #FEF2F2; color: #DC2626; font-size: 32px; font-weight: 800; line-height: 80px; margin: 0 auto 10px; border: 4px solid #FEE2E2; }
        
        .summary-card { flex: 3; min-width: 300px; background: white; padding: 24px; border-radius: 16px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); border: 1px solid #F3F4F6; }
        .summary-title { font-size: 14px; text-transform: uppercase; color: #6B7280; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 12px; }
        .summary-text { font-size: 15px; line-height: 1.6; color: #374151; }
        
        .clause-card { background: white; border-radius: 12px; margin-bottom: 0; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #E5E7EB; overflow: hidden; }
        .card-left-border-High { border-left: 6px solid #EF4444; }
        .card-left-border-Medium { border-left: 6px solid #F59E0B; }
        .card-left-border-Low { border-left: 6px solid #10B981; }
        
        .card-header { background: #F9FAFB; padding: 16px 24px; border-bottom: 1px solid #F3F4F6; display: flex; justify-content: space-between; align-items: center; }
        .clause-title { font-weight: 700; color: #111827; font-size: 18px;}
        .risk-badge { padding: 6px 14px; border-radius: 99px; font-size: 13px; font-weight: 700; letter-spacing: 0.5px; }
        .badge-High { background: #FEE2E2; color: #991B1B; }
        .badge-Medium { background: #FEF3C7; color: #92400E; }
        .badge-Low { background: #D1FAE5; color: #065F46; }
        
        .card-body { padding: 24px; }
        .quote-box { background: #F8FAFC; border-left: 4px solid #94A3B8; padding: 16px; color: #475569; font-family: 'Consolas', monospace; font-size: 14px; margin-bottom: 24px; line-height: 1.6; }
        
        .analysis-section { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; }
        
        .analysis-block { margin-bottom: 16px; }
        .analysis-label { font-size: 14px; font-weight: 700; color: #DC2626; margin-bottom: 6px; display: flex; align-items: center; gap: 6px; }
        .analysis-content { font-size: 15px; color: #374151; line-height: 1.7; text-align: justify; }
        
        .suggestion-box { background: #ECFDF5; border: 1px solid #A7F3D0; padding: 20px; border-radius: 10px; color: #065F46; font-size: 15px; line-height: 1.7; }
        .suggestion-label { font-size: 14px; font-weight: 700; color: #059669; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; }
        
        .law-box { margin-top: 16px; padding-top: 16px; border-top: 1px dashed #A7F3D0; font-size: 14px; color: #047857; }
        .law-label { font-weight: 700; margin-bottom: 4px; }
        
        .stButton button { margin-bottom: 20px; }
    </style>
    """)
    st.markdown(custom_css, unsafe_allow_html=True)

    # Back Button
    if back_callback:
        if st.button("← 返回工作台", key="back_btn_saas_fix_v12"):
            back_callback()
            st.rerun()

    # === 1. 创建标签页 ===
    tab_summary, tab_details = st.tabs(["📊 评估总览", "🔍 条款深度审查"])

    # === 2. 总览页 ===
    with tab_summary:
        # Calculate Stats
        high_count = len([x for x in structured_data if "高" in x.get('risk_level', '')])
        medium_count = len([x for x in structured_data if "中" in x.get('risk_level', '')])
        
        # Dashboard HTML
        score_html = f"""
        <div class="report-container">
            <div class="dashboard-row">
                <div class="score-card">
                    <div class="score-circle">{high_count}</div>
                    <div style="font-weight:600; color:#DC2626;">项核心风险</div>
                </div>
                <div class="summary-card" style="border:none; box-shadow:none;">
                    <div class="summary-title">✨ 风险分布</div>
                    <p style="font-size: 1.1rem; margin-top: 10px;">
                        <span style="color:#DC2626; font-weight:bold;">高风险：{high_count} 项</span> 
                        <span style="color:#D1D5DB; margin:0 10px;">|</span>
                        <span style="color:#D97706; font-weight:bold;">中风险：{medium_count} 项</span>
                        <span style="color:#D1D5DB; margin:0 10px;">|</span>
                        <span style="color:#059669; font-weight:bold;">低风险：{len(structured_data) - high_count - medium_count} 项</span>
                    </p>
                </div>
            </div>
        </div>
        """
        st.markdown(score_html, unsafe_allow_html=True)
        
        st.subheader("📝 执行摘要")
        # 使用原生滚动容器，限制高度
        with st.container(height=500, border=True):
            st.markdown(report_md)

    # === 3. 详情页 (折叠列表) ===
    with tab_details:
        st.info(f"共发现 {len(structured_data)} 处风险点，点击下方列表查看详情。")
        
        for idx, item in enumerate(structured_data):
            risk_level = item.get('risk_level', 'Low')
            clause_text = item.get('clause_text', '未识别到原文')
            risk_reason = item.get('risk_reason', '')
            deep_analysis = item.get('deep_analysis', '')
            law_content = item.get('law_content', '')
            law_reference = item.get('law_reference', '')
            dimension_id = item.get('dimension', '0')
            
            # Prepare Expander Title
            icon = "🔴" if "高" in risk_level else "🟢"
            clause_snippet = clause_text[:30] + "..." if len(clause_text) > 30 else clause_text
            expander_title = f"{icon} 【{risk_level}风险】条款 {idx+1}: {clause_snippet}"
            
            # Create Expander
            with st.expander(expander_title):
                # Prepare Card HTML
                risk_cls = "High" if "高" in risk_level else "Low"
                
                # Parse Analysis
                # 优先使用结构化字段，如果为空则尝试从文本解析
                structured_suggestion = item.get('suggestion')
                
                text_to_parse = deep_analysis if deep_analysis else risk_reason
                parsed = parse_deep_analysis(text_to_parse)
                
                violation_text = parsed["violation"] or risk_reason
                consequence_text = parsed["consequence"]
                
                # 逻辑修正：如果 structured_suggestion 存在且有效，直接使用
                # 否则尝试使用解析出的 suggestion，最后兜底
                if structured_suggestion and structured_suggestion.strip():
                    suggestion_text = structured_suggestion
                else:
                    suggestion_text = parsed["suggestion"] or "建议咨询专业律师。"
                
                violation_html = simple_md_to_html(violation_text)
                consequence_html = simple_md_to_html(consequence_text)
                suggestion_html = simple_md_to_html(suggestion_text)
                
                # Law HTML
                law_html = ""
                if law_content:
                    law_html = simple_md_to_html(law_content)
                elif law_reference and law_reference not in ["无", "None", "none"]:
                    law_html = f"涉及法条：{law_reference} (未检索到具体内容)"
                
                # Construct Columns
                left_col_html = ""
                
                # --- High Risk: Show Violation & Consequence ---
                if risk_cls == "High":
                    if violation_html:
                        left_col_html += f'<div class="analysis-block"><div class="analysis-label">⚠️ 违规点</div><div class="analysis-content">{violation_html}</div></div>'
                    if consequence_html:
                        left_col_html += f'<div class="analysis-block"><div class="analysis-label">💥 后果</div><div class="analysis-content">{consequence_html}</div></div>'
                
                # --- Low Risk: Show Analysis as Analysis ---
                else:
                    # Low risk uses deep_analysis as the main content
                    analysis_text = deep_analysis if deep_analysis else "条款内容未涉及典型风险点。"
                    left_col_html += f'<div class="analysis-block"><div class="analysis-label">📝 分析</div><div class="analysis-content">{simple_md_to_html(analysis_text)}</div></div>'

                    
                right_col_html = f"""
                <div class="suggestion-box">
                    <div class="suggestion-label">💡 修改建议</div>
                    <div>{suggestion_html}</div>
                    {f'<div class="law-box"><div class="law-label">⚖️ 法律依据：</div>{law_html}</div>' if law_html and risk_cls == "High" else ''}
                </div>
                """
                
                # 1. Generate Dimension Tags HTML
                # dim_tags = item.get("dimension_tags", []) # Deprecated, use dimension ID mapping
                tags_html = ""
                
                # Map Dimension ID to Name
                dim_name = DIM_MAP.get(str(dimension_id), "")
                if not dim_name and dimension_id and str(dimension_id) != "0":
                     # Try integer key if string key fails
                     dim_name = DIM_MAP.get(int(dimension_id), "")

                if dim_name:
                     tags_html += f'<span style="background:#EFF6FF; color:#1D4ED8; padding:2px 8px; border-radius:4px; font-size:12px; margin-right:5px; border:1px solid #DBEAFE;">{dim_name}</span>'
                elif risk_cls == "High":
                     # Fallback for High Risk if no dimension found
                     tags_html += f'<span style="background:#EFF6FF; color:#1D4ED8; padding:2px 8px; border-radius:4px; font-size:12px; margin-right:5px; border:1px solid #DBEAFE;">合规风险</span>'


                # 2. Construct Card Header
                risk_id = item.get('risk_id', '')
                risk_id_html = f'<span style="background:#F3F4F6; color:#4B5563; padding:2px 6px; border-radius:4px; font-size:11px; margin-left:8px; border:1px solid #E5E7EB;">ID: {risk_id}</span>' if risk_id else ""
                
                card_header = f"""
                <div class="card-header">
                    <span class="clause-title">条款 {idx + 1}</span>
                    <div style="display:flex; align-items:center; gap:10px;">
                        {tags_html}
                        <span class="risk-badge badge-{risk_cls}">{risk_cls} Risk</span>
                        {risk_id_html}
                    </div>
                </div>
                """

                card_html = f"""
                <div class="clause-card card-left-border-{risk_cls}" style="margin-bottom:0;">
                    {card_header}
                    <div class="card-body">
                        <div class="col-title" style="font-weight:bold; margin-bottom:10px;">原文全貌</div>
                        <div class="quote-box">{clause_text}</div>
                        <div class="analysis-section">
                            <div>{left_col_html}</div>
                            <div>{right_col_html}</div>
                        </div>
                    </div>
                </div>
                """
                
                st.markdown(card_html.replace("\n", ""), unsafe_allow_html=True)
