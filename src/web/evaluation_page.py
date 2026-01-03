"""
评测控制台页面 - 支持消融实验对比

功能：
1. 单模式评测
2. 多模式对比（消融实验）
3. 可视化评测结果
4. 历史结果查看
"""

import streamlit as st
import asyncio
import sys
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import ablation benchmark
try:
    from evaluation.ablation_benchmark import (
        run_ablation_benchmark,
        run_full_ablation_study,
        EvalMode
    )
    ABLATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ablation_benchmark: {e}")
    ABLATION_AVAILABLE = False

# Import legacy benchmark for backward compatibility
try:
    from evaluation.run_benchmark import run_benchmark
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False


def render_evaluation_page():
    """渲染评测控制台页面"""
    
    st.markdown("## 📊 消融实验评测控制台")
    st.markdown("> 对比不同配置下的模型表现，验证各组件的贡献度")
    
    # 选项卡
    tab1, tab2, tab3 = st.tabs(["🧪 消融实验", "📈 单模式评测", "📂 历史结果"])
    
    with tab1:
        render_ablation_study_tab()
    
    with tab2:
        render_single_mode_tab()
    
    with tab3:
        render_history_tab()


def render_ablation_study_tab():
    """消融实验标签页"""
    
    if not ABLATION_AVAILABLE:
        st.error("❌ 消融实验模块不可用，请检查 `evaluation/ablation_benchmark.py` 是否存在")
        return
    
    st.markdown("### 🔬 多模式对比实验")
    
    # 模式说明
    with st.expander("📖 模式说明", expanded=False):
        st.markdown("""
        | 模式 | 说明 | 组件 |
        |:----:|------|------|
        | **模式1** | 纯LLM | 无Prompt模板，直接输入条款 |
        | **模式2** | 基础Prompt | 有格式化Prompt，无规则引擎 |
        | **模式3** | 当前工作流 | Prompt + 规则引擎 |
        | **模式4** | 优化工作流 | 改进Prompt(CoT) + 规则引擎 |
        """)
    
    # 配置
    col1, col2, col3 = st.columns([2, 1, 1])
    
    # 可用数据集
    available_datasets = [
        "evaluation/llm_benchmark_dataset.json",
    ]
    
    with col1:
        data_path = st.selectbox(
            "测试数据集", 
            options=available_datasets,
            index=0,
            key="ablation_data_path"
        )
    
    with col2:
        limit = st.number_input(
            "样本数量限制", 
            min_value=1, 
            max_value=100, 
            value=5,
            key="ablation_limit"
        )
    
    with col3:
        source = st.selectbox(
            "LLM来源",
            options=["local", "cloud"],
            index=0,
            key="ablation_source"
        )
    
    # 模式选择
    st.markdown("**选择评测模式**（可多选）：")
    
    mode_cols = st.columns(4)
    modes_selected = []
    
    with mode_cols[0]:
        if st.checkbox("模式1: 纯LLM", value=True, key="mode1"):
            modes_selected.append(1)
    with mode_cols[1]:
        if st.checkbox("模式2: 基础Prompt", value=True, key="mode2"):
            modes_selected.append(2)
    with mode_cols[2]:
        if st.checkbox("模式3: 当前工作流", value=True, key="mode3"):
            modes_selected.append(3)
    with mode_cols[3]:
        if st.checkbox("模式4: 优化工作流", value=True, key="mode4"):
            modes_selected.append(4)
    
    # 开始按钮
    if st.button("🚀 开始消融实验", type="primary", use_container_width=True):
        if not modes_selected:
            st.warning("请至少选择一个评测模式")
            return
        
        if not Path(data_path).exists():
            st.error(f"数据文件不存在: {data_path}")
            return
        
        run_ablation_experiment(data_path, modes_selected, limit, source)
    
    # 显示结果
    if "ablation_results" in st.session_state:
        display_ablation_results(st.session_state.ablation_results)


def run_ablation_experiment(data_path: str, modes: list, limit: int, source: str):
    """运行消融实验"""
    
    st.markdown("### ⏳ 评测进度")
    
    progress_placeholder = st.empty()
    log_placeholder = st.empty()
    
    logs = []
    def log_callback(msg):
        logs.append(msg)
        log_placeholder.code("\n".join(logs[-10:]), language="text")
    
    all_results = {}
    total_modes = len(modes)
    
    for idx, mode in enumerate(modes):
        progress_placeholder.progress(
            (idx) / total_modes, 
            text=f"正在评测模式 {mode}: {EvalMode.name(mode)}..."
        )
        
        try:
            result = asyncio.run(
                run_ablation_benchmark(
                    data_path=data_path,
                    mode=mode,
                    limit=limit,
                    source=source,
                    log_callback=log_callback
                )
            )
            
            if result:
                all_results[f"mode_{mode}"] = result
                
        except Exception as e:
            st.error(f"模式 {mode} 评测失败: {e}")
    
    progress_placeholder.progress(1.0, text="评测完成！")
    
    if all_results:
        st.session_state.ablation_results = all_results
        st.success(f"✅ 消融实验完成！共评测 {len(all_results)} 种模式")
        
        # 保存结果
        save_ablation_results(all_results)
    else:
        st.error("❌ 所有模式评测失败")


def display_ablation_results(results: dict):
    """显示消融实验结果"""
    
    st.markdown("### 📊 评测结果对比")
    
    # 构建对比数据
    metrics_data = []
    
    for mode_key, mode_result in results.items():
        mode_num = int(mode_key.split("_")[1])
        metrics = mode_result.get("metrics", {})
        
        metrics_data.append({
            "模式": EvalMode.name(mode_num),
            "准确率": metrics.get("accuracy", 0),
            "F1分数": metrics.get("f1", 0),
            "精确率": metrics.get("precision", 0),
            "召回率": metrics.get("recall", 0),
            "解析成功率": metrics.get("parse_rate", 0),
            "幻觉率": metrics.get("hallucination_rate", 0),
        })
    
    df = pd.DataFrame(metrics_data)
    
    # 核心指标卡片
    st.markdown("#### 🎯 核心指标")
    
    cols = st.columns(len(results))
    for idx, (mode_key, mode_result) in enumerate(results.items()):
        mode_num = int(mode_key.split("_")[1])
        metrics = mode_result.get("metrics", {})
        
        with cols[idx]:
            st.markdown(f"**{EvalMode.name(mode_num)}**")
            st.metric("准确率", f"{metrics.get('accuracy', 0):.1%}")
            st.metric("F1分数", f"{metrics.get('f1', 0):.1%}")
            st.metric("幻觉率", f"{metrics.get('hallucination_rate', 0):.1%}", 
                     delta_color="inverse")
    
    # 对比表格
    st.markdown("#### 📋 详细指标对比")
    
    # 格式化百分比
    df_display = df.copy()
    for col in df_display.columns[1:]:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # 可视化图表
    st.markdown("#### 📈 可视化对比")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # 准确率 & F1 对比
        chart_data = df[["模式", "准确率", "F1分数"]].set_index("模式")
        st.bar_chart(chart_data)
        st.caption("准确率 & F1分数对比")
    
    with chart_col2:
        # 精确率 & 召回率 对比
        chart_data2 = df[["模式", "精确率", "召回率"]].set_index("模式")
        st.bar_chart(chart_data2)
        st.caption("精确率 & 召回率对比")
    
    # 详细结果展开
    st.markdown("#### 📝 详细评测记录")
    
    for mode_key, mode_result in results.items():
        mode_num = int(mode_key.split("_")[1])
        mode_name = EvalMode.name(mode_num)
        
        with st.expander(f"📋 {mode_name} - 详细结果"):
            display_single_mode_results(mode_result)


def display_single_mode_results(result: dict):
    """显示单个模式的详细结果"""
    
    items = result.get("results", [])
    
    if not items:
        st.info("无评测记录")
        return
    
    # 统计
    correct_count = sum(1 for item in items if item.get("correct_risk", False))
    st.markdown(f"**正确/总数**: {correct_count}/{len(items)}")
    
    # 逐条显示
    for item in items:
        is_correct = item.get("correct_risk", False)
        icon = "✅" if is_correct else "❌"
        
        pred = item.get("prediction", {})
        gt = item.get("ground_truth", {})
        
        with st.expander(f"{icon} Case {item.get('id', 'N/A')}: 预测={pred.get('risk_level', '?')} / 实际={gt.get('risk_level', '?')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🤖 模型预测**")
                st.markdown(f"- 风险等级: `{pred.get('risk_level', 'N/A')}`")
                st.markdown(f"- 证据: {pred.get('evidence', 'N/A')[:100] if pred.get('evidence') else 'N/A'}...")
                st.markdown(f"- 解析成功: {'✅' if pred.get('parse_success') else '❌'}")
                if pred.get('latency'):
                    st.markdown(f"- 响应时间: `{pred.get('latency', 0):.2f}s`")
            
            with col2:
                st.markdown("**📝 标准答案**")
                st.markdown(f"- 风险等级: `{gt.get('risk_level', 'N/A')}`")
                keywords = gt.get('reason_keywords', [])
                if keywords:
                    st.markdown(f"- 关键词: `{', '.join(keywords[:3])}`")
            
            st.markdown(f"**证据验证**: {'✅ 有效' if item.get('evidence_valid', True) else '❌ 幻觉'}")


def render_single_mode_tab():
    """单模式评测标签页"""
    
    st.markdown("### 🎯 单模式评测")
    
    if not ABLATION_AVAILABLE:
        st.error("❌ 评测模块不可用")
        return
    
    # 配置
    col1, col2, col3 = st.columns([2, 1, 1])
    
    # 可用数据集
    available_datasets = [
        "evaluation/llm_benchmark_dataset.json",
    ]
    
    with col1:
        data_path = st.selectbox(
            "测试数据集", 
            options=available_datasets,
            index=0,
            key="single_data_path"
        )
    
    with col2:
        mode = st.selectbox(
            "评测模式",
            options=[1, 2, 3, 4],
            format_func=lambda x: f"模式{x}: {EvalMode.name(x)}",
            key="single_mode"
        )
    
    with col3:
        limit = st.number_input(
            "样本限制", 
            min_value=1, 
            max_value=100, 
            value=10,
            key="single_limit"
        )
    
    source = st.radio(
        "LLM来源",
        options=["local", "cloud"],
        horizontal=True,
        key="single_source"
    )
    
    if st.button("▶️ 开始评测", type="primary", key="single_start"):
        if not Path(data_path).exists():
            st.error(f"数据文件不存在: {data_path}")
            return
        
        run_single_mode_evaluation(data_path, mode, limit, source)
    
    # 显示结果
    if "single_mode_result" in st.session_state:
        result = st.session_state.single_mode_result
        st.markdown(f"### 📊 {EvalMode.name(mode)} 评测结果")
        display_single_mode_results(result)


def run_single_mode_evaluation(data_path: str, mode: int, limit: int, source: str):
    """运行单模式评测"""
    
    log_placeholder = st.empty()
    
    logs = []
    def log_callback(msg):
        logs.append(msg)
        log_placeholder.code("\n".join(logs[-8:]), language="text")
    
    with st.spinner(f"正在评测模式 {mode}..."):
        try:
            result = asyncio.run(
                run_ablation_benchmark(
                    data_path=data_path,
                    mode=mode,
                    limit=limit,
                    source=source,
                    log_callback=log_callback
                )
            )
            
            if result:
                st.session_state.single_mode_result = result
                st.success("✅ 评测完成！")
            else:
                st.error("❌ 评测失败")
                
        except Exception as e:
            st.error(f"❌ 运行出错: {e}")


def render_history_tab():
    """历史结果标签页"""
    
    st.markdown("### 📂 历史评测结果")
    
    # 查找历史结果文件
    eval_dir = Path("evaluation")
    if not eval_dir.exists():
        st.info("暂无历史评测记录")
        return
    
    result_files = list(eval_dir.glob("ablation_results_*.json"))
    result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not result_files:
        st.info("暂无消融实验历史记录")
        return
    
    # 文件列表
    st.markdown(f"**找到 {len(result_files)} 个历史记录**")
    
    selected_file = st.selectbox(
        "选择结果文件",
        options=result_files,
        format_func=lambda x: x.name,
        key="history_file"
    )
    
    if selected_file:
        try:
            with open(selected_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            st.markdown(f"**文件**: `{selected_file.name}`")
            st.markdown(f"**创建时间**: {datetime.fromtimestamp(selected_file.stat().st_mtime)}")
            
            if st.button("📊 加载并显示", key="load_history"):
                st.session_state.ablation_results = data
                st.success("已加载历史结果")
                st.rerun()
            
            with st.expander("📄 原始JSON数据"):
                st.json(data)
                
        except Exception as e:
            st.error(f"加载失败: {e}")


def save_ablation_results(results: dict):
    """保存消融实验结果"""
    
    eval_dir = Path("evaluation")
    eval_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = eval_dir / f"ablation_results_{timestamp}.json"
    
    # 清理不可序列化的内容
    clean_results = {}
    for mode_key, mode_data in results.items():
        clean_results[mode_key] = {
            "mode": mode_data.get("mode"),
            "mode_name": mode_data.get("mode_name"),
            "metrics": mode_data.get("metrics", {}),
            "results": [
                {
                    "id": r.get("id"),
                    "correct_risk": r.get("correct_risk"),
                    "evidence_valid": r.get("evidence_valid"),
                    "prediction": r.get("prediction", {}),
                    "ground_truth": r.get("ground_truth", {}),
                }
                for r in mode_data.get("results", [])
            ]
        }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    st.info(f"💾 结果已保存至: {output_path}")
