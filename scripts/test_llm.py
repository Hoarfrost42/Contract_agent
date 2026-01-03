"""
LLM 诊断测试脚本
测试 Ollama 和 ChatOllama 的连接
"""
import requests
import sys

# 配置
BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen3:4b-instruct"

def test_ollama_direct():
    """测试 1: 直接 HTTP 调用 Ollama API"""
    print("=" * 50)
    print("测试 1: 直接 HTTP 调用 Ollama API")
    print("=" * 50)
    
    try:
        # 检查 Ollama 服务
        print(f"→ 检查 Ollama 服务: {BASE_URL}/api/tags")
        resp = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        print(f"  状态: {resp.status_code}")
        
        if resp.ok:
            models = [m['name'] for m in resp.json().get('models', [])]
            print(f"  可用模型: {models}")
            
            if MODEL_NAME not in models:
                print(f"  ⚠️ 警告: 目标模型 '{MODEL_NAME}' 不在列表中!")
                # 尝试找到匹配的模型
                for m in models:
                    if 'qwen' in m.lower():
                        print(f"  💡 建议使用: {m}")
        
        # 测试生成
        print(f"\n→ 测试生成请求: {BASE_URL}/api/generate")
        resp = requests.post(
            f"{BASE_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": "你好",
                "stream": False
            },
            timeout=120
        )
        print(f"  状态: {resp.status_code}")
        
        if resp.ok:
            data = resp.json()
            response_text = data.get('response', '')[:100]
            print(f"  响应 (前100字符): {response_text}")
            print("  ✅ 直接 HTTP 调用成功!")
            return True
        else:
            print(f"  ❌ 错误: {resp.text}")
            return False
            
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        return False

def test_chat_ollama():
    """测试 2: 通过 ChatOllama 调用"""
    print("\n" + "=" * 50)
    print("测试 2: 通过 ChatOllama (LangChain) 调用")
    print("=" * 50)
    
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage
        
        print(f"→ 初始化 ChatOllama: model={MODEL_NAME}")
        llm = ChatOllama(
            base_url=BASE_URL,
            model=MODEL_NAME,
            temperature=0.1,
            timeout=120,
        )
        
        print("→ 发送测试请求...")
        response = llm.invoke([HumanMessage(content="你好")])
        
        content = getattr(response, 'content', str(response))[:100]
        print(f"  响应 (前100字符): {content}")
        print("  ✅ ChatOllama 调用成功!")
        return True
        
    except Exception as e:
        print(f"  ❌ 异常: {type(e).__name__}: {e}")
        
        # 尝试获取更多错误信息
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()
        return False

def test_chat_ollama_sync():
    """测试 3: 使用同步方式调用 ChatOllama"""
    print("\n" + "=" * 50)
    print("测试 3: ChatOllama 使用 httpx 客户端")
    print("=" * 50)
    
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage
        import httpx
        
        print(f"→ 初始化 ChatOllama (使用 httpx)")
        
        # 创建自定义 httpx 客户端
        client = httpx.Client(timeout=120.0)
        
        llm = ChatOllama(
            base_url=BASE_URL,
            model=MODEL_NAME,
            temperature=0.1,
            client=client,
        )
        
        print("→ 发送测试请求...")
        response = llm.invoke([HumanMessage(content="你好")])
        
        content = getattr(response, 'content', str(response))[:100]
        print(f"  响应 (前100字符): {content}")
        print("  ✅ ChatOllama (httpx) 调用成功!")
        return True
        
    except Exception as e:
        print(f"  ❌ 异常: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    print("\n🔍 LLM 诊断测试\n")
    
    results = []
    
    # 测试 1
    results.append(("直接 HTTP", test_ollama_direct()))
    
    # 测试 2
    results.append(("ChatOllama", test_chat_ollama()))
    
    # 测试 3
    results.append(("ChatOllama (httpx)", test_chat_ollama_sync()))
    
    # 汇总
    print("\n" + "=" * 50)
    print("测试汇总")
    print("=" * 50)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
