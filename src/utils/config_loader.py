import os
from typing import Any, Dict, Optional

import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "config.yaml")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML configuration for the agent.

    Args:
        config_path: Optional path override for the config file.

    Returns:
        Parsed configuration dictionary.
    """

    path = config_path or DEFAULT_CONFIG_PATH
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Config file not found: {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data
