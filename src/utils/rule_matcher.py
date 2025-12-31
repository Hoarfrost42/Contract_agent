import json
import os
from typing import Optional, Dict, Any

def scan_text_for_risks(chunk_text: str) -> Optional[Dict[str, Any]]:
    """
    Scans the given text for risks based on keywords defined in risk_rules.json.

    Args:
        chunk_text: The text chunk to scan.

    Returns:
        The matching rule dictionary if a risk is found, otherwise None.
    """
    try:
        # Determine the path to risk_rules.json relative to this file
        # src/utils/rule_matcher.py -> src/core/risk_rules.json
        current_dir = os.path.dirname(os.path.abspath(__file__))
        rules_path = os.path.join(current_dir, '..', 'core', 'risk_rules.json')
        rules_path = os.path.normpath(rules_path)

        if not os.path.exists(rules_path):
            print(f"Error: Risk rules file not found at {rules_path}")
            return None

        with open(rules_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)

        for rule in rules:
            keywords = rule.get("keywords", [])
            for keyword in keywords:
                if keyword in chunk_text:
                    return rule

        return None

    except Exception as e:
        print(f"Error in scan_text_for_risks: {str(e)}")
        return None
