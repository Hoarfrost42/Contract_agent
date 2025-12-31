import yaml
import os
from pathlib import Path

def load_risk_standards(config_path: str = "configs/risk_standards.yaml") -> str:
    """
    Load risk standards from YAML and format them into a string for the system prompt.
    """
    # Resolve absolute path relative to project root
    project_root = Path(__file__).resolve().parents[2]
    full_path = project_root / config_path
    
    if not full_path.exists():
        return ""

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        general = data.get('general_criteria', {})
        dimensions = data.get('dimensions', {})
        
        # Format the output
        prompt_text = "请基于以下【八维风险判别标准】进行审查：\n\n"
        
        # Add Dimensions List
        prompt_text += "### 📐 八维评估体系 (Dimensions)\n"
        for key, value in dimensions.items():
            prompt_text += f"{key}. {value}\n"
        prompt_text += "\n"
        
        # Add General Criteria
        prompt_text += "### 🚦 风险评级标准 (Criteria)\n"
        if 'high' in general:
            prompt_text += f"{general['high']}\n"
        if 'medium' in general:
            prompt_text += f"{general['medium']}\n"
        if 'low' in general:
            prompt_text += f"{general['low']}\n"
            
        return prompt_text
        
    except Exception as e:
        print(f"Error loading risk standards: {e}")
        return ""

def get_risk_dimensions(config_path: str = "configs/risk_standards.yaml") -> dict:
    """
    Get the raw dimensions dictionary from the YAML file.
    """
    project_root = Path(__file__).resolve().parents[2]
    full_path = project_root / config_path
    
    if not full_path.exists():
        return {}

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data.get('dimensions', {})
    except Exception as e:
        print(f"Error loading risk dimensions: {e}")
        return {}
