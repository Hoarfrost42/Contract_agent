from langchain_core.tools import tool

from src.database.db_manager import query_law_exact


@tool
def lookup_law(law_name: str, article_id: str) -> str:
    """
    Use this tool to retrieve the exact content of a legal article. Input example:
    law_name='民法典', article_id='586'. Provide precise law names such as
    '中华人民共和国民法典' to guarantee accurate lookups.
    """

    content = query_law_exact(law_name, article_id)
    if content is None:
        return "未找到该法条，请确认法律名称和条号是否正确"
    return content
