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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 图表配置
CHART_CONFIG = {
    # 第一张图：基础评估指标 (Basic Metrics)
    "chart1_metrics": [
        {"key": "weighted_accuracy", "name": "加权准确率(非对称)", "format": "percent"},
        {"key": "macro_f1", "name": "Macro F1", "format": "percent"},
        {"key": "kappa_linear", "name": "Kappa (线性)", "format": "decimal_scaled"},
        {"key": "high_risk_f2", "name": "高风险 F2", "format": "percent"},
    ],
    # 第二张图：高级评估指标 (Advanced Metrics)
    "chart2_metrics": [
        {"key": "task_success_rate", "name": "任务成功率", "format": "percent"},
        {"key": "hallucination_rate", "name": "幻觉率", "format": "percent"},
        {"key": "rule_recall", "name": "规则召回率", "format": "percent"},
        {"key": "kappa_quadratic", "name": "Kappa (二次方)", "format": "decimal_scaled"},
    ],
    "mode_names": {
        1: "纯LLM",
        2: "基础Prompt",
        3: "当前工作流",
        4: "优化工作流",
    },
    "colors": ["#6366F1", "#8B5CF6", "#06B6D4", "#10B981"],  # 渐变色
}


def generate_report_charts(
    results: Dict[str, Any],
    output_dir: Path = None,
    timestamp: str = None
) -> List[str]:
    """
    生成三张核心图表：
    1. 基础评估指标 (Bar Chart)
    2. 高级评估指标 (Bar Chart)
    3. 混淆矩阵 (Heatmap Grid)
    
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
    
    chart_paths = []
    
    # ========== 图表1：基础评估指标 ==========
    metrics1 = CHART_CONFIG["chart1_metrics"]
    chart_path1 = _generate_bar_chart(
        results, modes, metrics1, 
        title="消融实验 - 基础评估指标 (Basic Metrics)",
        output_dir=output_dir,
        filename=f"chart1_basic_metrics_{timestamp}.png"
    )
    chart_paths.append(chart_path1)
    
    # ========== 图表2：高级评估指标 ==========
    metrics2 = CHART_CONFIG["chart2_metrics"]
    chart_path2 = _generate_bar_chart(
        results, modes, metrics2,
        title="消融实验 - 高级评估指标 (Advanced Metrics)",
        output_dir=output_dir,
        filename=f"chart2_advanced_metrics_{timestamp}.png"
    )
    chart_paths.append(chart_path2)
    
    # ========== 图表3：混淆矩阵 ==========
    chart_path3 = _generate_confusion_matrix_chart(
        results, modes,
        output_dir=output_dir,
        filename=f"chart3_confusion_matrix_{timestamp}.png"
    )
    chart_paths.append(chart_path3)
    
    return chart_paths


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
            
            # 处理不同格式：percent 和 decimal_scaled 都会乘以 100 进行绘制
            if metric_config["format"] == "percent":
                values.append(value * 100)
                labels.append(f'{value * 100:.1f}') # 不带%号，或者带？原图有%吗？annotate里自己加
            elif metric_config["format"] == "decimal_scaled":
                values.append(value * 100) # 放大100倍以便可视化
                labels.append(f'{value:.2f}') # 标签保持原始小数
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
    """生成混淆矩阵图表 (Grid)"""
    n_modes = len(modes)
    cols = min(n_modes, 2)
    rows = (n_modes + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_modes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    labels = ["高(H)", "中(M)", "低(L)"]
    
    for i, mode in enumerate(modes):
        ax = axes[i]
        mode_key = f"mode_{mode}"
        mode_name = CHART_CONFIG["mode_names"].get(mode, f"模式{mode}")
        
        if mode_key not in results or "metrics" not in results[mode_key]:
            ax.text(0.5, 0.5, "无数据", ha='center', va='center')
            continue
            
        # 获取并不是矩阵 (List[List[int]])
        conf_matrix = results[mode_key]["metrics"].get("conf_matrix", [[0]*3]*3)
        matrix = np.array(conf_matrix)
        
        # 绘制热力图
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.sum())
        
        # 添加数值标注
        for r_idx in range(3):
            for c_idx in range(3):
                val = matrix[r_idx, c_idx]
                total_in_row = matrix[r_idx].sum()
                percentage = val / total_in_row if total_in_row > 0 else 0
                
                # 字体颜色逻辑 (背景深色则白色，浅色则黑色)
                text_color = "white" if val > matrix.max() / 2 else "black"
                
                ax.text(c_idx, r_idx, f"{val}\n({percentage:.0%})", 
                        ha="center", va="center", color=text_color, fontweight='bold')
        
        ax.set_title(f"{mode_name}", fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("预测", fontsize=10)
        ax.set_ylabel("真实", fontsize=10)
        
        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
    fig.suptitle("消融实验 - 风险等级混淆矩阵 (High/Medium/Low)", fontsize=16, fontweight='bold', y=0.98)
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
