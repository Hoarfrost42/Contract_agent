import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


class DBManager:
    """Simple SQLite wrapper for storing law articles."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        root = Path(__file__).resolve().parents[2]
        default_path = root / "data" / "databases" / "laws.db"
        self.db_path = Path(db_path or default_path)
        if self.db_path.parent:
            os.makedirs(self.db_path.parent, exist_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_tables(self) -> None:
        """Create the law_articles table if it does not exist."""
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS law_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    law_name TEXT NOT NULL,
                    article_id TEXT NOT NULL,
                    content TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def insert_articles(self, articles: Iterable[Mapping[str, str]]) -> None:
        """Bulk insert article records."""
        records = [
            (article["law_name"], article["article_id"], article["content"])
            for article in articles
        ]
        if not records:
            return
        with self._get_conn() as conn:
            conn.executemany(
                """
                INSERT INTO law_articles (law_name, article_id, content)
                VALUES (?, ?, ?)
                """,
                records,
            )
            conn.commit()

    def clear_law(self, law_name: str) -> None:
        """Delete existing rows for a specific law."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM law_articles WHERE law_name = ?", (law_name,))
            conn.commit()

    def fetch_article(self, law_name: str, article_id: str) -> Optional[dict]:
        """Fetch a single article and return it as a dict."""
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT id, law_name, article_id, content
                FROM law_articles
                WHERE law_name LIKE ? AND article_id = ?
                """,
                (f"%{law_name}", article_id),
            ).fetchone()
            if row is None:
                return None
            return dict(row)

    def fetch_by_tag(self, tag: str) -> Optional[str]:
        """Fetch first article matching tag in content or tags."""
        with self._get_conn() as conn:
            try:
                # Try searching both content and tags
                cursor = conn.execute(
                    """
                    SELECT content
                    FROM law_articles
                    WHERE content LIKE ? OR tags LIKE ?
                    LIMIT 1
                    """,
                    (f"%{tag}%", f"%{tag}%"),
                )
            except sqlite3.OperationalError:
                # Fallback if tags column does not exist
                cursor = conn.execute(
                    """
                    SELECT content
                    FROM law_articles
                    WHERE content LIKE ?
                    LIMIT 1
                    """,
                    (f"%{tag}%",),
                )
            row = cursor.fetchone()
            if row:
                return row["content"]
            return None

    def get_law_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all laws."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT law_name, COUNT(*) as count
                FROM law_articles
                GROUP BY law_name
                ORDER BY count DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]


def query_law_exact(
    law_name: str, article_id: str, db_path: Optional[str] = None
) -> Optional[dict]:
    """Return the law record dict if it exists (with 'law_name' and 'content' fields)."""
    manager = DBManager(db_path)
    record = manager.fetch_article(law_name, article_id)
    return record


def query_law_by_tag(tag: str, db_path: Optional[str] = None) -> str:
    """
    Query law article by tag using fuzzy matching.
    Returns the content of the first matching article or a default message.
    """
    manager = DBManager(db_path)
    content = manager.fetch_by_tag(tag)
    if content:
        return content
    return "未检索到具体法条，建议参考民法典一般原则。"

