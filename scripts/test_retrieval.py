import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.database.db_manager import query_law_exact
import re

def test_retrieval():
    test_cases = [
        ("民法典", "1", True),
        ("中华人民共和国民法典", "1", True),
        ("民事诉讼法", "10", True), # Assuming Article 10 exists
        ("电子签名法", "1", True),
        ("不存在的法", "1", False),
    ]

    print("Testing Database Retrieval (Fuzzy Match)...")
    for law_name, article_id, expected in test_cases:
        content = query_law_exact(law_name, article_id)
        found = content is not None
        status = "PASS" if found == expected else "FAIL"
        print(f"[{status}] Query: '{law_name}' Article: {article_id} -> Found: {found}")
        if found:
             print(f"    Content snippet: {content[:20]}...")

    print("\nTesting Regex Pattern...")
    pattern = r"《?([\u4e00-\u9fa5]+?)》?第([0-9]+|[零一二三四五六七八九十百]+)条"
    regex_cases = [
        ("根据民法典第500条规定", "民法典", "500"),
        ("《中华人民共和国民事诉讼法》第21条", "中华人民共和国民事诉讼法", "21"),
        ("依据电子签名法第三条", "电子签名法", "三"),
        ("劳动合同法第10条", "劳动合同法", "10"),
    ]

    for text, expected_law, expected_id in regex_cases:
        match = re.search(pattern, text)
        if match:
            law = match.group(1)
            art = match.group(2)
            status = "PASS" if law == expected_law and art == expected_id else "FAIL"
            print(f"[{status}] Text: '{text}' -> Law: '{law}', Art: '{art}'")
        else:
            print(f"[FAIL] Text: '{text}' -> No match")

if __name__ == "__main__":
    test_retrieval()
