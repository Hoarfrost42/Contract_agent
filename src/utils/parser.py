import re
import json
from typing import List


def split_contract(text: str) -> List[str]:
    """
    使用正则将合同文本切分为有意义的条款。
    
    支持格式:
    - 第X条 / 第X章
    - 1. / 1、 / (1) / （1）
    - 1.1
    """
    if not text:
        return []

    # 条款标题正则
    # 1. "第...条" 或 "第...章" (行首)
    # 2. "1." 或 "1、" 或 "(1)" 或 "（1）" (行首)
    # 3. "1.1" (行首)
    header_pattern = r"(?:^|\n)\s*(?:第[0-9零一二三四五六七八九十百]+[条章]|[\(（]?[0-9]+[\)）]?[\.、]?|[0-9]+\.[0-9]+)\s*"
    
    # 使用 finditer 找到所有标题及其位置
    matches = list(re.finditer(header_pattern, text))
    
    if not matches:
        # 如果未发现结构，回退到按行切分
        return [line.strip() for line in text.splitlines() if len(line.strip()) >= 5]

    clauses = []
    start_idx = 0
    
    # 如果第一个匹配项不是在开头，捕获前言部分
    if matches[0].start() > 0:
        preamble = text[:matches[0].start()].strip()
        if len(preamble) >= 5:
            clauses.append(preamble)
            
    for i, match in enumerate(matches):
        current_start = match.start()
        # 结束位置是下一个匹配项的开始，或者是文本末尾
        next_start = matches[i+1].start() if i + 1 < len(matches) else len(text)
        
        clause_content = text[current_start:next_start].strip()
        
        # 清理：移除条款内部多余的换行符
        clause_content = re.sub(r'\n\s*', ' ', clause_content)
        
        if len(clause_content) >= 5:
            clauses.append(clause_content)
            
    return clauses


def chunk_contract(text: str, chunk_size: int = 5) -> List[str]:
    """
    将条款聚合为 JSON 分块 (已弃用，保留用于兼容性或参考)。
    
    参数:
        text: 合同全文
        chunk_size: 每个分块包含的条款数 (默认 5)
        
    返回:
        JSON 字符串列表，每个代表一个条款列表。
        示例: ['["clause1", "clause2", ...]', ...]
    """

    clauses = split_contract(text)
    if not clauses:
        return []

    chunks: List[str] = []
    
    # 按 chunk_size 批量处理
    for i in range(0, len(clauses), chunk_size):
        batch = clauses[i : i + chunk_size]
        
        # 序列化为 JSON 字符串
        # ensure_ascii=False 保持中文可读
        json_chunk = json.dumps(batch, ensure_ascii=False)
        chunks.append(json_chunk)

    return chunks
