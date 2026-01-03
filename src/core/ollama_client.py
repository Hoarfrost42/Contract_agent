"""
统一的 Ollama API 客户端模块

提供同步和异步接口调用 Ollama API，供以下模块统一调用：
- engine.py (实际工作流)
- ablation_benchmark.py (评测脚本)
- llm.py (LLM 客户端)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    统一的 Ollama API 客户端
    
    使用 requests 直接调用 Ollama API，避免 httpx 兼容性问题。
    
    使用方式：
        client = OllamaClient(base_url="http://localhost:11434", model="qwen3:4b")
        response = client.chat("你好")
        
        # 异步调用
        response = await client.achat("你好")
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434", 
        model: str = "qwen3:4b-instruct", 
        temperature: float = 0
    ):
        """
        初始化 Ollama 客户端
        
        Args:
            base_url: Ollama API 地址
            model: 模型名称
            temperature: 温度参数（0=确定性输出）
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
    
    def chat(self, prompt: str, timeout: int = 120) -> str:
        """
        发送聊天请求（同步）
        
        Args:
            prompt: 用户输入
            timeout: 超时时间（秒）
            
        Returns:
            模型输出文本
        """
        import requests
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": self.temperature}
                },
                timeout=timeout
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except requests.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise RuntimeError(f"Ollama API error: {e}")
    
    async def achat(self, prompt: str, timeout: int = 120) -> str:
        """
        发送聊天请求（异步）
        
        使用 run_in_executor 将同步调用包装为异步
        
        Args:
            prompt: 用户输入
            timeout: 超时时间（秒）
            
        Returns:
            模型输出文本
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(prompt, timeout))
    
    def generate(self, prompt: str, timeout: int = 120) -> str:
        """
        使用 /api/generate 接口（无对话记忆）
        
        Args:
            prompt: 输入文本
            timeout: 超时时间（秒）
            
        Returns:
            模型输出文本
        """
        import requests
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature}
                },
                timeout=timeout
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise RuntimeError(f"Ollama API error: {e}")


# ============================================================================
# 便捷函数
# ============================================================================

_default_client: Optional[OllamaClient] = None


def get_ollama_client(
    base_url: str = None, 
    model: str = None, 
    temperature: float = 0
) -> OllamaClient:
    """
    获取 Ollama 客户端（支持单例模式）
    
    Args:
        base_url: Ollama API 地址，默认从配置读取
        model: 模型名称，默认从配置读取
        temperature: 温度参数
        
    Returns:
        OllamaClient 实例
    """
    global _default_client
    
    # 如果提供了参数，创建新客户端
    if base_url or model:
        from src.utils.config_loader import load_config
        config = load_config()
        llm_config = config.get("llm_config", {})
        
        return OllamaClient(
            base_url=base_url or llm_config.get("base_url", "http://localhost:11434"),
            model=model or llm_config.get("model_name", "qwen3:4b-instruct"),
            temperature=temperature
        )
    
    # 否则使用默认单例
    if _default_client is None:
        from src.utils.config_loader import load_config
        config = load_config()
        llm_config = config.get("llm_config", {})
        
        _default_client = OllamaClient(
            base_url=llm_config.get("base_url", "http://localhost:11434"),
            model=llm_config.get("model_name", "qwen3:4b-instruct"),
            temperature=0
        )
    
    return _default_client
