from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import requests
import streamlit as st

# Custom imports
from src.utils.file_loader import extract_text_from_file
from src.database.db_manager import DBManager
from src.utils.prompt_manager import get_risk_dimensions
from src.web.report_renderer import render_saas_report
from src.web.evaluation_page import render_evaluation_page
from src.utils.report_generator import generate_word_report

# ============================================================================
# 1. Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Contract Risk AI",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 2. Apple Human Interface & Glassmorphism CSS
# ============================================================================
def inject_custom_css():
    st.markdown("""
    <style>
        /* Font Imports */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@600;700&display=swap');
        
        /* 1. Reset */
        .stApp {
            background-color: #FAFAFA;
            /* Soft animated mesh gradient */
            background-image: 
                radial-gradient(at 0% 0%, rgba(200, 200, 255, 0.4) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(220, 200, 255, 0.4) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(200, 220, 255, 0.4) 0px, transparent 50%),
                radial-gradient(at 0% 100%, rgba(230, 240, 255, 0.4) 0px, transparent 50%);
            background-attachment: fixed;
            background-size: cover;
        }
        
        * { font-family: 'Inter', sans-serif; }
        
        /* 2. Elegant Typography */
        h1.hero-title {
            font-family: 'Playfair Display', serif;
            font-size: 3.5rem;
            font-weight: 700;
            color: #1A1A1A;
            text-align: center;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
            text-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        p.hero-subtitle {
            font-family: 'Inter', sans-serif;
            font-size: 1.1rem;
            color: #666;
            text-align: center;
            margin-bottom: 3rem;
            font-weight: 300;
        }

        /* 3. Centered Hero Glass Card */
        .hero-glass-container {
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(24px) saturate(180%);
            -webkit-backdrop-filter: blur(24px) saturate(180%);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.6);
            box-shadow: 
                0 20px 40px rgba(0, 0, 0, 0.04),
                0 1px 3px rgba(0, 0, 0, 0.02),
                inset 0 1px 1px rgba(255, 255, 255, 0.6);
            padding: 40px;
            max-width: 800px;
            margin: 0 auto;
        }

        /* 4. iOS Segmented Control Override */
        /* Hacking Streamlit's Radio to look like iOS Segmented Control */
        div.row-widget.stRadio > div {
            flex-direction: row;
            background-color: rgba(118, 118, 128, 0.12);
            padding: 3px;
            border-radius: 9px;
            width: fit-content;
            margin: 0 auto 30px auto;
            border: none;
        }
        
        div.row-widget.stRadio > div > label {
            background-color: transparent;
            border-radius: 7px;
            padding: 6px 20px;
            margin: 0;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 0.9rem;
            font-weight: 500;
            color: #444;
            border: none;
        }
        
        /* Selected State Hack - difficult in pure CSS as Streamlit controls internal state 
           But we can style the active look roughly or rely on Streamlit's internal classes if consistent */
        div.row-widget.stRadio > div > label[data-baseweb="radio"] {
             /* Reset baseweb styles */
        }
        
        /* 5. Clean Sidebar */
        section[data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.5);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.3);
        }

        /* 6. Primary Action Button */
        div.stButton > button[kind="primary"] {
            width: 100%;
            border-radius: 14px;
            height: 50px;
            font-size: 1.1rem;
            background: #111; /* Stark black for contrast or Apple Blue */
            color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin-top: 20px;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #333;
            transform: translateY(-1px);
        }
        
    </style>
    """, unsafe_allow_html=True)
    
# 3. Helper Functions
API_BASE_URL = "http://127.0.0.1:8000"

def submit_job(text: str, risk_dims: List[str], use_cloud: bool, enable_deep_reflection: bool) -> Dict[str, Any]:
    try:
        payload = {
            "contract_text": text,
            "risk_dimensions": risk_dims,
            "use_cloud_model": use_cloud,
            "enable_deep_reflection": enable_deep_reflection
        }
        resp = requests.post(f"{API_BASE_URL}/submit", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

def fetch_progress(job_id: str) -> Dict[str, Any]:
    try:
        resp = requests.get(f"{API_BASE_URL}/progress/{job_id}", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "logs": [str(e)]}

# 4. View: Dashboard
def view_dashboard():
    # Hero Titles
    st.markdown('<h1 class="hero-title">Contract AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Intelligent Legal Risk Analysis & Review</p>', unsafe_allow_html=True)
    
    # Centered Container
    st.markdown('<div class="hero-glass-container">', unsafe_allow_html=True)
    
    # Use Radio as Segmented Control (Centered)
    # We use columns to center it because Streamlit aligns left by default
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        input_method = st.radio(
            "Input Method",
            ["📄 Upload File", "✍️ Paste Text"],
            horizontal=True,
            label_visibility="collapsed",
            key="input_method_switch"
        )
    
    contract_text = ""
    
    # Toggle Content
    if input_method == "📄 Upload File":
        uploaded_file = st.file_uploader(
            "Drag & Drop your contract here", 
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            with st.spinner("Processing file..."):
                try:
                    contract_text = extract_text_from_file(uploaded_file)
                    st.success(f"File loaded: {uploaded_file.name} ({len(contract_text)} chars)")
                except Exception as e:
                    st.error(f"Error: {e}")
                    
    else:  # Paste Text
        contract_text = st.text_area(
            "Contract Text", 
            height=300, 
            placeholder="Paste your legal document content here...",
            label_visibility="collapsed"
        )

    # Sidebar Settings (Cleaned Up)
    with st.sidebar:
        st.title("Settings")
        
        st.markdown("### Model")
        use_cloud = st.toggle("Use Cloud Model (DeepSeek)", False)
        
        with st.expander("Advanced Options", expanded=False):
            enable_reflection = st.toggle("Deep Reflection", True)
            st.markdown("**Risk Dimensions**")
            risk_map = get_risk_dimensions()
            selected_dims = []
            for risk_id, desc in risk_map.items():
                if st.checkbox(f"{risk_id}: {desc}", value=True):
                    selected_dims.append(risk_id)
    
    # Primary Action Button (Inside the glass card, full width)
    if contract_text:
        # Check if selected_dims is configured (defaults if sidebar not rendered yet?)
        # We need to access the sidebar values. Streamlit handles this via session state keys implicitly or we read them.
        # But since they are in sidebar, they are accessible.
        
        # NOTE: Since selected_dims is local in sidebar block, we need to default it if empty.
        # Ideally we'd store it in session state, but for simplicty we'll just re-read or assume all if None.
        if 'selected_dims' not in locals():
            # If default run, use all
             selected_dims = list(get_risk_dimensions().keys())

        if st.button("Analyze Contract", type="primary"):
            if not selected_dims:
                st.warning("Please select at least one risk dimension in Advanced Options.")
            else:
                with st.spinner("Initiating analysis..."):
                    # sidebar vars need to be accessed carefully if scopes are different, but Streamlit rerun logic usually holds them.
                    # We re-fetch or use defaults
                    res = submit_job(contract_text, selected_dims, use_cloud, enable_reflection if 'enable_reflection' in locals() else True)
                    if "job_id" in res:
                        st.session_state.current_job_id = res["job_id"]
                        st.session_state.start_time = time.time()
                        st.session_state.page = "report"
                        st.rerun()
                    else:
                        st.error(f"Submission failed: {res.get('error')}")

    st.markdown('</div>', unsafe_allow_html=True) # End glass container

# 5. View: Report & Progress
def view_report():
    job_id = st.session_state.get("current_job_id")
    if not job_id:
        st.warning("无活跃任务")
        if st.button("返回首页"):
            st.session_state.page = "dashboard"
            st.rerun()
        return

    # Header
    col_head_1, col_head_2 = st.columns([3, 1])
    with col_head_1:
         st.markdown(f"### 🔍 正在分析 (Job: `{job_id[-6:]}`)")
    with col_head_2:
        if st.button("❌ 停止/返回", kind="secondary"):
             st.session_state.page = "dashboard"
             st.rerun()

    # Progress Container (Glass)
    progress_placeholder = st.empty()
    log_placeholder = st.empty()
    result_placeholder = st.empty()
    
    # Polling Loop
    while True:
        status_data = fetch_progress(job_id)
        status = status_data.get("status", "unknown")
        logs = status_data.get("logs", [])
        
        # Calculate Progress
        total_steps = 15 # Estimated
        current_step = len(logs)
        progress_val = min(current_step / total_steps, 0.95)
        if status == "done": progress_val = 1.0
        
        with progress_placeholder.container():
            st.markdown(f'<div class="glass-card">', unsafe_allow_html=True)
            st.progress(progress_val)
            st.caption(f"Status: {status.upper()} | Elapsed: {int(time.time() - st.session_state.start_time)}s")
            st.markdown('</div>', unsafe_allow_html=True)

        with log_placeholder.container():
            if logs:
                st.code(logs[-1], language="text")

        if status == "done":
            result = status_data.get("result", {})
            st.success("分析完成！")
            
            # --- RENDER REPORT ---
            render_saas_report(result)
            
            # Download Word
            if result:
                docx_io = generate_word_report(result)
                st.download_button(
                    "📥 下载 Word 报告",
                    data=docx_io,
                    file_name=f"Risk_Report_{job_id}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    type="primary"
                )
            break
        
        elif status == "failed":
            st.error("分析任务失败")
            break
            
        time.sleep(1)

# 6. Main Router
def main():
    inject_custom_css()
    
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"
    
    # Sidebar Navigation using Radio (Styled as pills ideally, but standard for now)
    with st.sidebar:
        st.title("Contract AI")
        page_nav = st.radio("导航", ["分析台", "模型评估"], 
            index=0 if st.session_state.page != "evaluation" else 1)
        
        if page_nav == "模型评估" and st.session_state.page != "evaluation":
            st.session_state.page = "evaluation"
            st.rerun()
        elif page_nav == "分析台" and st.session_state.page == "evaluation":
            st.session_state.page = "dashboard"
            st.rerun()

    if st.session_state.page == "dashboard":
        view_dashboard()
    elif st.session_state.page == "report":
        view_report()
    elif st.session_state.page == "evaluation":
        render_evaluation_page()

if __name__ == "__main__":
    main()
