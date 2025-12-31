"""
合并 LLM 生成的测试用例为 Benchmark 数据集
"""
import json
from pathlib import Path
from datetime import datetime

def main():
    base_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent
    
    # 加载所有测试用例
    files = ['GENERAL.json', 'LABOR.json', 'LEASE.json', 'SALES.json']
    all_cases = []
    
    for fname in files:
        fpath = base_dir / fname
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read().rstrip()
            if content.endswith('.'):
                content = content[:-1]
            cases = json.loads(content)
            
            # 添加来源标记
            for c in cases:
                c['source'] = 'llm_generated'
                c['source_file'] = fname
                c['generated_at'] = datetime.now().isoformat()
            
            all_cases.extend(cases)
            print(f'加载 {fname}: {len(cases)} 条')
    
    # 统计
    from collections import Counter
    types = Counter(c['case_type'] for c in all_cases)
    print(f'\n总计: {len(all_cases)} 条测试用例')
    print(f'  正例: {types.get("positive", 0)}')
    print(f'  负例: {types.get("negative", 0)}')
    print(f'  边界: {types.get("boundary", 0)}')
    
    # 保存为 JSONL 格式
    output_file = output_dir / 'llm_benchmark_dataset.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for case in all_cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
    
    print(f'\n已保存至: {output_file}')
    
    # 同时保存一份 JSON 格式（方便查看）
    json_file = output_dir / 'llm_benchmark_dataset.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)
    
    print(f'已保存至: {json_file}')

if __name__ == '__main__':
    main()
