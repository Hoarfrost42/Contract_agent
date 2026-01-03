"""
数据合并脚本：将 GENERAL/LABOR/LEASE/SALES.json 合并为 llm_benchmark_dataset.json

功能：
1. 合并四个 JSON 文件
2. 添加 source_domain 字段
3. ID 去重检查
4. 数据质量验证
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_json(path: Path) -> List[Dict[str, Any]]:
    """加载 JSON 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_item(item: Dict[str, Any], idx: int, source: str) -> List[str]:
    """验证单个样本的数据质量
    
    Returns:
        List[str]: 错误信息列表，空表示无错误
    """
    errors = []
    
    # 必填字段检查
    required_fields = ['id', 'contract_text', 'case_type']
    for field in required_fields:
        if not item.get(field):
            errors.append(f"[{source}] 样本 {idx}: 缺少必填字段 '{field}'")
    
    # case_type 值检查
    valid_case_types = {'high_positive', 'medium_positive', 'negative'}
    case_type = item.get('case_type', '')
    if case_type and case_type not in valid_case_types:
        errors.append(f"[{source}] 样本 {idx}: case_type 值无效 '{case_type}'，应为 {valid_case_types}")
    
    # expected_risks 结构检查（对于正例）
    if case_type in {'high_positive', 'medium_positive'}:
        expected_risks = item.get('expected_risks', [])
        if not expected_risks:
            errors.append(f"[{source}] 样本 {idx}: 正例样本缺少 expected_risks")
        else:
            for i, risk in enumerate(expected_risks):
                if not isinstance(risk, dict):
                    errors.append(f"[{source}] 样本 {idx}: expected_risks[{i}] 应为字典")
                elif not risk.get('risk_id'):
                    errors.append(f"[{source}] 样本 {idx}: expected_risks[{i}] 缺少 risk_id")
    
    return errors


def merge_datasets(root_dir: Path, output_path: Path) -> Dict[str, Any]:
    """合并数据集
    
    Returns:
        Dict with 'success', 'total', 'errors', 'duplicates'
    """
    source_files = ['GENERAL.json', 'LABOR.json', 'LEASE.json', 'SALES.json']
    domain_map = {
        'GENERAL.json': 'GENERAL',
        'LABOR.json': 'LABOR',
        'LEASE.json': 'LEASE',
        'SALES.json': 'SALES',
    }
    
    merged = []
    seen_ids = set()
    duplicates = []
    all_errors = []
    
    for filename in source_files:
        filepath = root_dir / filename
        if not filepath.exists():
            print(f"⚠️ 文件不存在: {filepath}")
            continue
        
        print(f"📂 加载 {filename}...")
        items = load_json(filepath)
        domain = domain_map[filename]
        
        for idx, item in enumerate(items):
            # 添加 source_domain
            item['source_domain'] = domain
            
            # 数据质量验证
            errors = validate_item(item, idx, filename)
            all_errors.extend(errors)
            
            # ID 去重检查
            item_id = item.get('id', '')
            if item_id in seen_ids:
                duplicates.append(item_id)
                print(f"⚠️ 重复 ID: {item_id}")
            else:
                seen_ids.add(item_id)
                merged.append(item)
        
        print(f"   ✅ 加载 {len(items)} 条样本")
    
    # 输出统计
    print(f"\n📊 合并统计:")
    print(f"   总样本数: {len(merged)}")
    print(f"   重复 ID: {len(duplicates)}")
    print(f"   数据质量错误: {len(all_errors)}")
    
    if all_errors:
        print("\n⚠️ 数据质量问题:")
        for err in all_errors[:10]:  # 只显示前 10 条
            print(f"   {err}")
        if len(all_errors) > 10:
            print(f"   ... 还有 {len(all_errors) - 10} 条错误")
    
    # 写入输出文件
    if not all_errors:  # 仅无错误时写入
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 已保存到: {output_path}")
    else:
        print(f"\n❌ 存在数据质量问题，未保存文件")
    
    return {
        'success': len(all_errors) == 0,
        'total': len(merged),
        'errors': all_errors,
        'duplicates': duplicates,
    }


if __name__ == '__main__':
    # 项目根目录
    root_dir = Path(__file__).resolve().parents[1]
    output_path = Path(__file__).resolve().parent / 'llm_benchmark_dataset.json'
    
    print("=" * 60)
    print("📦 消融实验数据集合并脚本")
    print("=" * 60)
    
    result = merge_datasets(root_dir, output_path)
    
    if result['success']:
        print(f"\n🎉 合并成功！共 {result['total']} 条样本")
    else:
        print(f"\n💥 合并失败，请修复数据质量问题")
