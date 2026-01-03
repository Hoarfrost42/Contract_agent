import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.database.db_manager import DBManager

ARTICLE_PATTERN = re.compile(r"(第[零一二三四五六七八九十百千]+条)\s+(.*)")


CHINESE_DIGITS: Dict[str, int] = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "两": 2,
}
UNIT_MAP: Dict[str, int] = {"十": 10, "百": 100, "千": 1000}
SECTION_UNIT_MAP: Dict[str, int] = {"万": 10_000, "亿": 100_000_000}


def chinese_numeral_to_str(text: str) -> str:
    """Convert Chinese numerals (e.g., '五百八十六') into arabic string."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty Chinese numeral.")
    if cleaned.isdigit():
        return cleaned

    total = 0
    section = 0
    number = 0

    for char in cleaned:
        if char in CHINESE_DIGITS:
            number = CHINESE_DIGITS[char]
        elif char in UNIT_MAP:
            unit = UNIT_MAP[char]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
        elif char in SECTION_UNIT_MAP:
            section += number
            if section == 0:
                section = 1
            total += section * SECTION_UNIT_MAP[char]
            section = 0
            number = 0
        else:
            # Skip unsupported characters silently.
            continue

    total += section + number
    return str(total)


def arabic_to_chinese(num: int) -> str:
    """Convert arabic integers into Chinese numerals (supports up to 4 digits)."""
    if num == 0:
        return "零"

    digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    units = ["", "十", "百", "千"]
    result = ""
    num_str = str(num)
    length = len(num_str)
    zero_flag = False

    for idx, char in enumerate(num_str):
        digit = int(char)
        pos = length - idx - 1
        if digit == 0:
            zero_flag = True
            continue

        if zero_flag:
            result += "零"
            zero_flag = False

        if digit == 1 and pos == 1 and result == "":
            result += units[pos]
        else:
            result += digits[digit] + units[pos]

    return result or digits[0]


def parse_articles(text: str, law_name: str) -> List[Dict[str, str]]:
    """Parse the raw text file into article records."""
    articles: List[Dict[str, str]] = []
    current_article: Optional[Dict[str, str]] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        normalized_line = re.sub(
            r"(第[零一二三四五六七八九十百千]+条)", r"\1 ", line, count=1
        )
        match = ARTICLE_PATTERN.search(normalized_line)
        if match:
            if current_article:
                articles.append(current_article)

            heading = match.group(1)
            body = match.group(2).strip()
            numeral = heading.replace("第", "").replace("条", "")
            article_id = chinese_numeral_to_str(numeral)
            current_article = {
                "law_name": law_name,
                "article_id": article_id,
                "content": body,
            }
        elif current_article:
            current_article["content"] = (
                f"{current_article['content']}\n{line}".strip()
            )

    if current_article:
        articles.append(current_article)

    return articles


def load_text(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find law text at {file_path}")
    return file_path.read_text(encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    db_path = root / "data" / "databases" / "laws.db"
    
    # List of law files to ingest
    law_files = [
        "中华人民共和国民法典.txt",
        "中华人民共和国民事诉讼法.txt",
        "中华人民共和国电子签名法.txt",
        "中华人民共和国劳动合同法.txt"
    ]
    
    db_manager = DBManager(str(db_path))
    db_manager.create_tables()

    for filename in law_files:
        text_path = root / "Evaldata" / "Raw_datasets" / filename
        law_name = filename.replace(".txt", "")
        
        print(f"Processing {law_name}...")
        try:
            raw_text = load_text(text_path)
            articles = parse_articles(raw_text, law_name)
            
            # Clear existing data for this law to avoid duplicates
            db_manager.clear_law(law_name)
            db_manager.insert_articles(articles)
            print(f"  - Inserted {len(articles)} articles.")
        except FileNotFoundError:
            print(f"  - File not found: {text_path}")
        except Exception as e:
            print(f"  - Error processing {law_name}: {e}")

    # Verification: Check one article from Electronic Signature Law
    sample_law = "中华人民共和国电子签名法"
    sample_article_id = "1"
    sample = db_manager.fetch_article(sample_law, sample_article_id)
    
    print("\nVerification:")
    if sample:
        article_num_cn = arabic_to_chinese(int(sample_article_id))
        print(f"《{sample_law}》第{article_num_cn}条 {sample['content']}")
    else:
        print(f"Article {sample_article_id} of {sample_law} not found in database.")


if __name__ == "__main__":
    main()
