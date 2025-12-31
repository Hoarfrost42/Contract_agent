"""
消融实验图表生成模块
生成评测指标的柱状图并保存为 PNG 文件
"""

import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# 使用非交互式后端
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 图表配置
CHART_CONFIG = {
    "metrics": [
        {"key": "accuracy", "name": "准确率 (Accuracy)", "format": "percent"},
        {"key": "f1", "name": "F1 分数", "format": "percent"},
        {"key": "precision", "name": "精确率 (Precision)", "format": "percent"},
        {"key": "recall", "name": "召回率 (Recall)", "format": "percent"},
        {"key": "hallucination_rate", "name": "幻觉率 (Hallucination)", "format": "percent"},
    ],
    "mode_names": {
        1: "纯LLM",
        2: "基础Prompt",
        3: "当前工作流",
        4: "优化工作流",
    },
    "colors": ["#6366F1", "#8B5CF6", "#06B6D4", "#10B981"],  # 渐变色
}


def generate_ablation_charts(
    results: Dict[str, Any],
    output_dir: Path = None,
    timestamp: str = None
) -> List[str]:
    """
    根据消融实验结果生成柱状图
    
    Args:
        results: 消融实验结果字典，格式为 {"mode_1": {...}, "mode_2": {...}, ...}
        output_dir: 输出目录，默认为 evaluation/charts/
        timestamp: 时间戳，用于文件命名
    
    Returns:
        生成的图表文件路径列表
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "charts"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    chart_paths = []
    
    # 提取模式和数据
    modes = sorted([int(k.split("_")[1]) for k in results.keys() if k.startswith("mode_")])
    if not modes:
        print("警告: 没有找到有效的评测结果")
        return []
    
    # 为每个指标生成一张图
    for metric_config in CHART_CONFIG["metrics"]:
        metric_key = metric_config["key"]
        metric_name = metric_config["name"]
        is_percent = metric_config["format"] == "percent"
        
        # 收集数据
        values = []
        labels = []
        for mode in modes:
            mode_key = f"mode_{mode}"
            if mode_key in results and "metrics" in results[mode_key]:
                value = results[mode_key]["metrics"].get(metric_key, 0)
                values.append(value * 100 if is_percent else value)
                labels.append(CHART_CONFIG["mode_names"].get(mode, f"模式{mode}"))
        
        if not values:
            continue
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制柱状图
        bars = ax.bar(
            range(len(values)),
            values,
            color=CHART_CONFIG["colors"][:len(values)],
            edgecolor="white",
            linewidth=1.5,
            width=0.6,
        )
        
        # 在柱子上方显示数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{value:.1f}%' if is_percent else f'{value:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold',
                color='#374151',
            )
        
        # 设置标签和标题
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("百分比 (%)" if is_percent else "数值", fontsize=11)
        ax.set_title(f"消融实验 - {metric_name}", fontsize=14, fontweight='bold', pad=15)
        
        # 设置 Y 轴范围
        if is_percent:
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 100)
        
        # 添加网格线
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # 美化边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E5E7EB')
        ax.spines['bottom'].set_color('#E5E7EB')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        chart_filename = f"ablation_{metric_key}_{timestamp}.png"
        chart_path = output_dir / chart_filename
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        chart_paths.append(str(chart_path))
        print(f"📊 已生成图表: {chart_path}")
    
    return chart_paths


def generate_combined_chart(
    results: Dict[str, Any],
    output_dir: Path = None,
    timestamp: str = None
) -> str:
    """
    生成综合对比图（所有指标在一张图上）
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
        return ""
    
    # 准备数据
    metrics = CHART_CONFIG["metrics"]
    x = range(len(metrics))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 为每个模式绘制一组柱子
    for i, mode in enumerate(modes):
        mode_key = f"mode_{mode}"
        if mode_key not in results or "metrics" not in results[mode_key]:
            continue
        
        values = []
        for metric_config in metrics:
            value = results[mode_key]["metrics"].get(metric_config["key"], 0)
            values.append(value * 100)  # 转为百分比
        
        offset = (i - len(modes) / 2 + 0.5) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            values,
            width,
            label=CHART_CONFIG["mode_names"].get(mode, f"模式{mode}"),
            color=CHART_CONFIG["colors"][i % len(CHART_CONFIG["colors"])],
        )
    
    # 设置标签
    ax.set_xticks(x)
    ax.set_xticklabels([m["name"] for m in metrics], fontsize=10)
    ax.set_ylabel("百分比 (%)", fontsize=11)
    ax.set_title("消融实验 - 综合指标对比", fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 110)
    
    # 美化
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # 保存
    chart_filename = f"ablation_combined_{timestamp}.png"
    chart_path = output_dir / chart_filename
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"📊 已生成综合图表: {chart_path}")
    return str(chart_path)


if __name__ == "__main__":
    # 测试用例
    test_results = {
        "mode_1": {"metrics": {"accuracy": 0.6, "f1": 0.55, "precision": 0.7, "recall": 0.45, "hallucination_rate": 0.3}},
        "mode_2": {"metrics": {"accuracy": 0.7, "f1": 0.65, "precision": 0.75, "recall": 0.55, "hallucination_rate": 0.2}},
        "mode_3": {"metrics": {"accuracy": 0.8, "f1": 0.75, "precision": 0.85, "recall": 0.65, "hallucination_rate": 0.1}},
        "mode_4": {"metrics": {"accuracy": 0.85, "f1": 0.8, "precision": 0.88, "recall": 0.72, "hallucination_rate": 0.08}},
    }
    
    charts = generate_ablation_charts(test_results)
    combined = generate_combined_chart(test_results)
    print(f"\n生成了 {len(charts)} 张单指标图表和 1 张综合图表")
