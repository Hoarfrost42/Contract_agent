"""
验证 LLM 生成的 Benchmark 测试用例质量
"""
import json
from collections import Counter
from pathlib import Path

def main():
    base_dir = Path(__file__).parent.parent
    
    # 加载规则库获取有效 ID
    with open(base_dir / 'src/core/risk_rules.json', 'r', encoding='utf-8') as f:
        rules = json.load(f)
    valid_ids = {r['risk_id'] for r in rules}
    rule_names = {r['risk_id']: r['risk_name'] for r in rules}
    print(f'规则库有效 ID 共 {len(valid_ids)} 个')

    # 检查每个文件
    files = ['GENERAL.json', 'LABOR.json', 'LEASE.json', 'SALES.json']
    all_cases = []
    stats = {'total': 0, 'positive': 0, 'negative': 0, 'boundary': 0}
    invalid_ids = set()
    issues = []
    covered_rules = set()

    for fname in files:
        fpath = base_dir / fname
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read().rstrip()
                # 移除末尾可能的句点
                if content.endswith('.'):
                    content = content[:-1]
                cases = json.loads(content)
            
            print(f'\n=== {fname} ===')
            print(f'  条目数: {len(cases)}')
            
            case_types = Counter(c.get('case_type') for c in cases)
            print(f'  类型分布: positive={case_types.get("positive", 0)}, negative={case_types.get("negative", 0)}, boundary={case_types.get("boundary", 0)}')
            
            for c in cases:
                stats['total'] += 1
                ct = c.get('case_type', 'unknown')
                if ct in stats:
                    stats[ct] += 1
                
                # 检查 risk_id 有效性
                for risk in c.get('expected_risks', []):
                    rid = risk.get('risk_id')
                    if rid:
                        covered_rules.add(rid)
                        if rid not in valid_ids:
                            invalid_ids.add(rid)
                            issues.append(f"{c['id']}: 无效 risk_id '{rid}'")
                
                all_cases.append(c)
        except json.JSONDecodeError as e:
            print(f'  ❌ JSON 解析错误: {e}')
        except Exception as e:
            print(f'  ❌ 读取错误: {e}')

    print(f'\n{"="*50}')
    print(f'=== 汇总统计 ===')
    print(f'{"="*50}')
    print(f'总条目: {stats["total"]}')
    print(f'  - 正例: {stats.get("positive", 0)}')
    print(f'  - 负例: {stats.get("negative", 0)}')
    print(f'  - 边界: {stats.get("boundary", 0)}')
    print(f'\n规则覆盖率: {len(covered_rules)}/{len(valid_ids)} ({100*len(covered_rules)/len(valid_ids):.1f}%)')

    if invalid_ids:
        print(f'\n⚠️ 发现无效 risk_id: {sorted(invalid_ids)}')
        print(f'\n问题详情:')
        for issue in issues:
            print(f'  - {issue}')
        return False
    else:
        print(f'\n✅ 所有 risk_id 均有效')
    
    # 检查未覆盖的规则
    uncovered = valid_ids - covered_rules
    if uncovered:
        print(f'\n⚠️ 未覆盖的规则 ({len(uncovered)} 条):')
        for rid in sorted(uncovered):
            print(f'  - {rid}: {rule_names.get(rid, "?")}')
    
    return True

if __name__ == '__main__':
    main()
