"""
消融实验结果融合工具

用于将不同实验的 Mode 1-3 结果与单独运行的 Mode 4 结果融合，并生成图表。
这样可以复用已有的 Mode 1-3 数据，只需重新运行 Mode 4 实验。

使用示例:
    python evaluation/merge_results.py \
        --base /path/to/full_results.json \
        --mode4 /path/to/mode4_only_results.json \
        --output /path/to/merged_results
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def load_json(path: str) -> Dict[str, Any]:
    """加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str):
    """保存 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def merge_results(
    base_path: str,
    mode4_path: str,
    output_dir: str = None
) -> str:
    """
    融合实验结果
    
    Args:
        base_path: 包含 Mode 1-3 的完整结果 JSON 文件路径
        mode4_path: 仅包含 Mode 4 的结果 JSON 文件路径
        output_dir: 输出目录（默认与 mode4_path 同目录）
    
    Returns:
        融合后的 JSON 文件路径
    """
    print(f"📖 加载基础结果（Mode 1-3）: {base_path}")
    base_data = load_json(base_path)
    
    print(f"📖 加载 Mode 4 结果: {mode4_path}")
    mode4_data = load_json(mode4_path)
    
    # 验证数据结构
    if "mode_1" not in base_data or "mode_2" not in base_data or "mode_3" not in base_data:
        raise ValueError("基础结果文件必须包含 mode_1, mode_2, mode_3")
    
    if "mode_4" not in mode4_data:
        raise ValueError("Mode 4 结果文件必须包含 mode_4")
    
    # 融合结果
    merged = {
        "mode_1": base_data["mode_1"],
        "mode_2": base_data["mode_2"],
        "mode_3": base_data["mode_3"],
        "mode_4": mode4_data["mode_4"],
        # 添加元数据
        "_metadata": {
            "merged_at": datetime.now().isoformat(),
            "base_source": base_path,
            "mode4_source": mode4_path,
            "base_timestamp": base_data.get("_metadata", {}).get("timestamp", "unknown"),
            "mode4_timestamp": mode4_data.get("_metadata", {}).get("timestamp", "unknown"),
        }
    }
    
    # 确定输出目录
    if output_dir is None:
        output_dir = Path(mode4_path).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存融合结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_path = output_dir / f"merged_results_{timestamp}.json"
    save_json(merged, str(merged_path))
    print(f"✅ 融合结果已保存: {merged_path}")
    
    return str(merged_path)


def generate_charts(json_path: str):
    """调用图表生成器生成图表"""
    from chart_generator import generate_report_charts
    
    data = load_json(json_path)
    output_dir = Path(json_path).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    charts = generate_report_charts(data, output_dir, timestamp)
    print(f"\n📊 图表生成完成！共 {len(charts)} 张")
    for chart in charts:
        print(f"   - {chart}")


def main():
    parser = argparse.ArgumentParser(
        description="融合消融实验结果并生成图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法：融合 Mode 1-3 和 Mode 4 结果并生成图表
  python merge_results.py --base full_results.json --mode4 mode4_results.json

  # 指定输出目录
  python merge_results.py --base full_results.json --mode4 mode4_results.json --output ./merged

  # 仅融合不生成图表
  python merge_results.py --base full_results.json --mode4 mode4_results.json --no-charts
        """
    )
    
    parser.add_argument(
        "--base", "-b",
        required=True,
        help="包含 Mode 1-3 的完整结果 JSON 文件路径"
    )
    parser.add_argument(
        "--mode4", "-m",
        required=True,
        help="仅包含 Mode 4 的结果 JSON 文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出目录（默认与 Mode 4 结果同目录）"
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="仅融合结果，不生成图表"
    )
    
    args = parser.parse_args()
    
    # 验证文件存在
    if not Path(args.base).exists():
        print(f"❌ 基础结果文件不存在: {args.base}")
        return
    if not Path(args.mode4).exists():
        print(f"❌ Mode 4 结果文件不存在: {args.mode4}")
        return
    
    # 融合结果
    try:
        merged_path = merge_results(args.base, args.mode4, args.output)
        
        # 生成图表
        if not args.no_charts:
            generate_charts(merged_path)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ 处理失败: {e}")


if __name__ == "__main__":
    main()
