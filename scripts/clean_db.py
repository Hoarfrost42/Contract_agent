import sqlite3
from pathlib import Path

def clean_db():
    db_path = Path("data/databases/laws.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check counts before
    cursor.execute("SELECT law_name, COUNT(*) FROM law_articles GROUP BY law_name")
    print("Before:", cursor.fetchall())
    
    # Delete '民法典'
    cursor.execute("DELETE FROM law_articles WHERE law_name = '民法典'")
    conn.commit()
    
    # Check counts after
    cursor.execute("SELECT law_name, COUNT(*) FROM law_articles GROUP BY law_name")
    print("After:", cursor.fetchall())
    
    conn.close()

if __name__ == "__main__":
    clean_db()
