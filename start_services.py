from __future__ import annotations

# 抑制 jieba 库中 pkg_resources 弃用警告
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

import os
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread
from typing import Any, List, TextIO

from src.core.llm import LLMClient


def _stream_output(pipe: Any, target: TextIO) -> None:
    """Read binary output from pipe, decode as UTF-8, and write to target."""
    try:
        # iter(pipe.readline, b"") works for binary streams
        for line in iter(pipe.readline, b""):
            decoded_line = line.decode("utf-8", errors="replace")
            target.write(decoded_line)
            target.flush()
    finally:
        pipe.close()


def spawn_process(command: List[str], cwd: Path) -> subprocess.Popen:
    env = os.environ.copy()
    # Force child process to output UTF-8
    env["PYTHONIOENCODING"] = "utf-8"
    
    # Use binary mode (text=False) to avoid subprocess trying to decode with system locale (GBK)
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,  # BINARY MODE
        env=env,
    )
    
    if proc.stdout:
        Thread(target=_stream_output, args=(proc.stdout, sys.stdout), daemon=True).start()
    if proc.stderr:
        Thread(target=_stream_output, args=(proc.stderr, sys.stderr), daemon=True).start()
    
    return proc


def main() -> None:
    root = Path(__file__).resolve().parent
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]
    frontend_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/web/app.py",
        "--server.port",
        "9000",
        "--server.address",
        "127.0.0.1",
    ]

    backend_proc = frontend_proc = None
    try:
        print("正在启动后端 FastAPI 服务 (http://127.0.0.1:8000)...")
        backend_proc = spawn_process(backend_cmd, root)
        time.sleep(3)
        print("正在启动前端 Streamlit 应用...")
        frontend_proc = spawn_process(frontend_cmd, root)
        print("服务已就绪，请访问 http://localhost:9000")
        while True:
            time.sleep(1)
            
            # Check backend
            backend_ret = backend_proc.poll()
            if backend_ret is not None:
                raise RuntimeError(f"后端服务意外退出 (退出码: {backend_ret})。")
            
            # Check frontend
            frontend_ret = frontend_proc.poll()
            if frontend_ret is not None:
                if frontend_ret == 0:
                    print("\n✅ 用户已在前端触发系统关闭。")
                    break
                else:
                    raise RuntimeError(f"前端应用意外退出 (退出码: {frontend_ret})。")
    except KeyboardInterrupt:
        print("\n收到中断信号，正在关闭服务...")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        for proc, name in [(frontend_proc, "前端"), (backend_proc, "后端")]:
            if proc and proc.poll() is None:
                print(f"正在停止{name}进程...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"{name}进程未响应，强制结束。")
                    proc.kill()
        
        # Unload LLM
        try:
            print("正在释放大模型显存...")
            LLMClient().unload_model()
        except Exception as e:
            print(f"释放显存失败: {e}")


if __name__ == "__main__":
    main()
