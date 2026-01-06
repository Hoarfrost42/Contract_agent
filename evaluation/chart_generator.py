"""
消融实验图表生成模块
生成评测指标的柱状图及混淆矩阵，并保存为 PNG 文件
"""

import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

# 使用非交互式后端
matplotlib.use('Agg')

# 设置中文字体（Linux/Windows 兼容）
# 按优先级设置字体：Linux 云端 -> Windows
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 图表配置 (根据展示策略优化)
CHART_CONFIG = {
    # 图表2：核心性能指标 (The Safety & Logic Bar) - 必须展示的"SOTA 证据"
    "chart2_metrics": [
        {"key": "high_risk_f2", "name": "High-Risk F2", "format": "percent"},  # 主指标：安全性
        {"key": "kappa_quadratic", "name": "Quadratic Kappa", "format": "decimal_scaled"},  # 逻辑指标
        {"key": "weighted_accuracy", "name": "Weighted Accuracy", "format": "percent"},  # 落地指标
        {"key": "risk_id_precision", "name": "Risk ID Precision", "format": "percent"},  # 可信度指标
    ],
    # 图表3：系统稳定性指标 (The Quality Check) - 证明"证据准入"机制的有效性
    "chart3_metrics": [
        {"key": "hallucination_rate", "name": "幻觉率 ↓", "format": "percent_inverse"},  # 越低越好
        {"key": "task_success_rate", "name": "任务成功率", "format": "percent"},
        {"key": "high_risk_leakage", "name": "高风险漏判率 ↓", "format": "percent_inverse"},  # 自定义指标
    ],
    "mode_names": {
        1: "纯LLM",
        2: "基础Prompt", 
        3: "当前工作流",
        4: "优化工作流",
    },
    "colors": ["#EF4444", "#F59E0B", "#06B6D4", "#10B981"],  # 红-橙-青-绿 渐变
}


def generate_report_charts(
    results: Dict[str, Any],
    output_dir: Path = None,
    timestamp: str = None
) -> List[str]:
    """
    生成三张核心图表（根据展示策略优化）：
    1. 混淆矩阵 (The Behavior Map) - 4模式并排对比
    2. 核心性能指标 (The Safety & Logic Bar) 
    3. 系统稳定性指标 (The Quality Check)
    
    Returns:
        生成的图表路径列表
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "charts"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 提取模式
    modes = sorted([int(k.split("_")[1]) for k in results.keys() if k.startswith("mode_")])
    if not modes:
        print("警告: 没有找到有效的评测结果")
        return []
    
    # 预处理：计算 high_risk_leakage (高风险漏判率)
    _add_high_risk_leakage(results, modes)
    
    chart_paths = []
    
    # ========== 图表1：混淆矩阵 (The Behavior Map) ==========
    chart_path1 = _generate_confusion_matrix_chart(
        results, modes,
        output_dir=output_dir,
        filename=f"chart1_confusion_matrix_{timestamp}.png"
    )
    chart_paths.append(chart_path1)
    
    # ========== 图表2：核心性能指标 (The Safety & Logic Bar) ==========
    metrics2 = CHART_CONFIG["chart2_metrics"]
    chart_path2 = _generate_bar_chart(
        results, modes, metrics2,
        title="核心性能指标 (The Safety & Logic Bar)",
        output_dir=output_dir,
        filename=f"chart2_performance_{timestamp}.png"
    )
    chart_paths.append(chart_path2)
    
    # ========== 图表3：系统稳定性指标 (The Quality Check) ==========
    metrics3 = CHART_CONFIG["chart3_metrics"]
    chart_path3 = _generate_bar_chart(
        results, modes, metrics3,
        title="系统稳定性指标 (The Quality Check)",
        output_dir=output_dir,
        filename=f"chart3_quality_{timestamp}.png"
    )
    chart_paths.append(chart_path3)
    
    return chart_paths


def _add_high_risk_leakage(results: Dict[str, Any], modes: List[int]):
    """计算并添加 high_risk_leakage 指标（高风险漏判率 = High→Medium 比例）"""
    for mode in modes:
        mode_key = f"mode_{mode}"
        if mode_key not in results or "metrics" not in results[mode_key]:
            continue
        
        conf_matrix = results[mode_key]["metrics"].get("conf_matrix", [[0]*3]*3)
        # High = row 0, Medium = col 1
        # High→Medium = conf_matrix[0][1]
        high_to_medium = conf_matrix[0][1]
        total_high = sum(conf_matrix[0])  # 真实高风险总数
        
        # 高风险漏判率 = High→Medium / Total High
        leakage_rate = high_to_medium / total_high if total_high > 0 else 0
        results[mode_key]["metrics"]["high_risk_leakage"] = leakage_rate


def _generate_bar_chart(
    results: Dict[str, Any],
    modes: List[int],
    metrics: List[Dict],
    title: str,
    output_dir: Path,
    filename: str
) -> str:
    """生成柱状图"""
    x = range(len(metrics))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 为每个模式绘制一组柱子
    for i, mode in enumerate(modes):
        mode_key = f"mode_{mode}"
        if mode_key not in results or "metrics" not in results[mode_key]:
            continue
        
        values = []
        labels = []
        for metric_config in metrics:
            value = results[mode_key]["metrics"].get(metric_config["key"], 0)
            fmt = metric_config["format"]
            
            # 处理不同格式
            if fmt == "percent":
                values.append(value * 100)
                labels.append(f'{value * 100:.1f}%')
            elif fmt == "percent_inverse":  # 越低越好（如幻觉率）
                values.append(value * 100)
                labels.append(f'{value * 100:.1f}%')
            elif fmt == "decimal_scaled":
                values.append(value * 100)  # 放大100倍以便可视化
                labels.append(f'{value:.2f}')  # 标签保持原始小数
            else:
                values.append(value)
                labels.append(f'{value:.2f}')
        
        offset = (i - len(modes) / 2 + 0.5) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            values,
            width,
            label=CHART_CONFIG["mode_names"].get(mode, f"模式{mode}"),
            color=CHART_CONFIG["colors"][i % len(CHART_CONFIG["colors"])],
            edgecolor="white",
            linewidth=1
        )
        
        # 在柱子上添加数值标签
        for bar, label_text in zip(bars, labels):
            height = bar.get_height()
            ax.annotate(label_text,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='#374151')
    
    # 设置标签
    ax.set_xticks(x)
    ax.set_xticklabels([m["name"] for m in metrics], fontsize=11, fontweight='bold')
    ax.set_ylabel("数值 (已归一化到 0-100)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10, frameon=False)
    
    # 动态设置 Y 轴上限
    ax.set_ylim(0, 115) 
    
    # 美化
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E5E7EB')
    ax.spines['bottom'].set_color('#E5E7EB')
    
    plt.tight_layout()
    
    # 保存
    chart_path = output_dir / filename
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"📊 已生成图表: {chart_path}")
    return str(chart_path)


def _generate_confusion_matrix_chart(
    results: Dict[str, Any],
    modes: List[int],
    output_dir: Path,
    filename: str
) -> str:
    """生成混淆矩阵图表 (The Behavior Map) - 4模式并排对比"""
    n_modes = len(modes)
    
    # 强制使用 1 行 x N 列布局（并排对比）
    fig, axes = plt.subplots(1, n_modes, figsize=(4.5 * n_modes, 4.5))
    if n_modes == 1:
        axes = [axes]
    else:
        axes = list(axes)
        
    labels = ["高", "中", "低"]
    
    for i, mode in enumerate(modes):
        ax = axes[i]
        mode_key = f"mode_{mode}"
        mode_name = CHART_CONFIG["mode_names"].get(mode, f"模式{mode}")
        
        if mode_key not in results or "metrics" not in results[mode_key]:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center')
            continue
            
        conf_matrix = results[mode_key]["metrics"].get("conf_matrix", [[0]*3]*3)
        matrix = np.array(conf_matrix)
        total = matrix.sum() if matrix.sum() > 0 else 1
        
        # 绘制热力图
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
        
        # 添加数值标注 + 视觉焦点
        for r_idx in range(3):
            for c_idx in range(3):
                val = matrix[r_idx, c_idx]
                total_in_row = matrix[r_idx].sum()
                percentage = val / total_in_row if total_in_row > 0 else 0
                
                # 字体颜色逻辑
                text_color = "white" if val > matrix.max() / 2 else "black"
                
                # 特殊标记关键区域
                cell_text = f"{val}\n({percentage:.0%})"
                fontweight = 'bold'
                fontsize = 9
                
                # 视觉焦点：High→High (对角线) / Medium→High (防御升级) / High→Medium (漏判)
                if r_idx == 0 and c_idx == 0:  # High→High (正确召回)
                    fontsize = 10
                elif r_idx == 1 and c_idx == 0:  # Medium→High (防御性升级)
                    fontsize = 10
                elif r_idx == 0 and c_idx == 1:  # High→Medium (漏判！)
                    cell_text = f"{val}\n({percentage:.0%})\n[!]"
                    fontsize = 10
                
                ax.text(c_idx, r_idx, cell_text, 
                        ha="center", va="center", color=text_color, 
                        fontweight=fontweight, fontsize=fontsize)
        
        ax.set_title(f"{mode_name}", fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("预测", fontsize=10)
        if i == 0:
            ax.set_ylabel("真实", fontsize=10)
            
    fig.suptitle("行为映射矩阵 (The Behavior Map) - 展示\"单向狙击\"战术效果", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    chart_path = output_dir / filename
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"📊 已生成图表: {chart_path}")
    return str(chart_path)


if __name__ == "__main__":
    import argparse
    import json
    import os
    
    parser = argparse.ArgumentParser(description="生成消融实验图表")
    parser.add_argument("json_file", nargs="?", help="消融实验结果 JSON 文件路径")
    parser.add_argument("--output", "-o", help="图表输出目录（默认：与 JSON 文件同目录）")
    
    args = parser.parse_args()
    
    if args.json_file and os.path.exists(args.json_file):
        print(f"📖 读取结果文件: {args.json_file}")
        try:
            with open(args.json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 使用 JSON 文件的时间戳或当前时间
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 确定输出目录
            if args.output:
                output_dir = Path(args.output)
            else:
                output_dir = Path(args.json_file).parent
                
            charts = generate_report_charts(data, output_dir, timestamp)
            print(f"\n✅ 图表生成完成！共 {len(charts)} 张，保存在: {output_dir.absolute()}")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌生成失败: {e}")
    else:
        print("⚠️ 未提供有效的 JSON 文件路径")
