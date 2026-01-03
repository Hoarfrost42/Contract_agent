import re
import sys
from pathlib import Path
from typing import List, Dict

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.database.db_manager import DBManager
from src.utils.text_utils import chinese_numeral_to_str

def parse_processed_line(line: str) -> Dict[str, str]:
    """
    Parse a line from processed law file.
    Format: 《LawName》ArticleNumber规定，Content
    Example: 《中华人民共和国劳动法》第一条规定，为了...
    """
    # Regex to capture LawName, ArticleNumber (Chinese), and Content
    # ArticleNumber is like "第一条" or "第一百二十条"
    # Content is everything after "规定，"
    
    # Note: process_law_text.py output: f"《{law_name}》{header}规定，{content}\n"
    # header is "第X条"
    
    pattern = re.compile(r"《(.*?)》(第[零一二三四五六七八九十百千]+条)规定，(.*)")
    match = pattern.match(line.strip())
    
    if not match:
        return None
        
    law_name = match.group(1)
    article_cn = match.group(2) # e.g. "第一条"
    content = match.group(3)
    
    # Convert article_cn to arabic string
    numeral_str = article_cn.replace("第", "").replace("条", "")
    try:
        article_id = chinese_numeral_to_str(numeral_str)
    except ValueError:
        print(f"Warning: Could not convert numeral {numeral_str}")
        article_id = numeral_str # Fallback
        
    return {
        "law_name": law_name,
        "article_id": article_id,
        "content": content
    }

def import_laws(processed_dir: str, db_path: str):
    db_manager = DBManager(db_path)
    db_manager.create_tables()
    
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        print(f"Error: Processed directory not found: {processed_dir}")
        return

    total_inserted = 0
    
    for file_path in processed_path.glob("*.txt"):
        print(f"Importing: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        articles = []
        current_law_name = ""
        
        for line in lines:
            record = parse_processed_line(line)
            if record:
                articles.append(record)
                current_law_name = record['law_name']
        
        if articles:
            # Clear existing entries for this law to avoid duplicates
            # Assuming all articles in file belong to same law (which they do based on filename/process logic)
            # But let's be safe and use the law_name from the first record
            if current_law_name:
                db_manager.clear_law(current_law_name)
                
            db_manager.insert_articles(articles)
            print(f"  - Inserted {len(articles)} articles for {current_law_name}")
            total_inserted += len(articles)
            
    print(f"Total articles inserted: {total_inserted}")

if __name__ == "__main__":
    PROCESSED_DIR = r"F:\Agent_back\11.29\Evaldata\processed_laws"
    DB_PATH = r"F:\Agent_back\11.29\data\databases\laws.db"
    
    import_laws(PROCESSED_DIR, DB_PATH)
