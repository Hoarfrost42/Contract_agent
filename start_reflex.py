"""
Contract AI 一键启动脚本
同时启动 FastAPI 后端 (port 8002) 和 Reflex 前端 (port 3000)
支持 Ctrl+C 完全终止所有进程（包括子进程）
"""
import subprocess
import signal
import sys
import os
import time

# 目录配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REFLEX_DIR = os.path.join(PROJECT_ROOT, "reflex_web")

# 端口配置
BACKEND_PORT = 8002  # FastAPI 后端
FRONTEND_PORT = 3000  # Reflex 前端

# 进程列表
processes = []
# 退出标志
should_exit = False


def kill_process_tree(pid):
    """
    使用 taskkill 强制终止进程及其所有子进程 (Windows)
    这比 terminate() 更可靠，能确保所有子进程都被清理
    """
    try:
        # /T = 终止子进程, /F = 强制终止
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            timeout=10
        )
    except Exception as e:
        print(f"   警告: 清理 PID {pid} 时出错: {e}")


def cleanup():
    """清理所有子进程及其子进程树"""
    global processes
    print("\n\n🛑 正在停止所有服务...")
    
    # 收集所有 PID
    pids = []
    for p in processes:
        if p.poll() is None:  # 进程仍在运行
            pids.append(p.pid)
    
    # 使用 taskkill 强制终止进程树
    for pid in pids:
        print(f"   终止进程树: PID {pid}")
        kill_process_tree(pid)
    
    # 等待短暂时间确保清理完成
    time.sleep(1)
    
    # 额外清理：查找并终止可能残留的 node.exe 进程
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq node.exe", "/FO", "CSV"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "node.exe" in result.stdout:
            print("   发现残留 node 进程，正在清理...")
            subprocess.run(["taskkill", "/F", "/IM", "node.exe"], capture_output=True, timeout=5)
    except:
        pass
    
    print("✅ 所有服务已停止")
    print("💡 终端保持运行，您可以继续输入命令。\n")


def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    global should_exit
    should_exit = True


def main():
    global processes, should_exit
    
    print("=" * 60)
    print("🚀 Contract AI 启动脚本 (增强版)")
    print("=" * 60)
    print(f"📂 项目目录: {PROJECT_ROOT}")
    print("💡 按 Ctrl+C 可完全停止所有服务")
    print("=" * 60 + "\n")
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 1. 启动 FastAPI 后端
        print(f"🔧 启动 FastAPI 后端 (port {BACKEND_PORT})...")
        backend = subprocess.Popen(
            [sys.executable, "-m", "src.api.main"],
            cwd=PROJECT_ROOT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # Windows: 创建新进程组
        )
        processes.append(backend)
        time.sleep(3)
        
        if backend.poll() is not None:
            print("❌ FastAPI 后端启动失败！")
            cleanup()
            return
        print(f"✅ FastAPI 后端已启动: http://localhost:{BACKEND_PORT}\n")
        
        # 2. 启动 Reflex 前端 (不使用 shell=True，直接调用)
        print(f"🌐 启动 Reflex 前端 (port {FRONTEND_PORT})...")
        
        # 查找 reflex 可执行文件路径
        reflex_cmd = "reflex"
        if sys.platform == "win32":
            # 尝试在虚拟环境中查找
            venv_reflex = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "reflex.exe")
            if os.path.exists(venv_reflex):
                reflex_cmd = venv_reflex
        
        frontend = subprocess.Popen(
            [reflex_cmd, "run"],
            cwd=REFLEX_DIR,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # Windows: 创建新进程组
        )
        processes.append(frontend)
        print(f"✅ Reflex 前端启动中: http://localhost:{FRONTEND_PORT}\n")
        
        print("=" * 60)
        print("🎉 所有服务已启动！")
        print(f"   - 前端: http://localhost:{FRONTEND_PORT}")
        print(f"   - 后端: http://localhost:{BACKEND_PORT}")
        print("   - 按 Ctrl+C 完全停止所有服务")
        print("=" * 60 + "\n")
        
        # 等待前端就绪后自动打开浏览器
        print("⏳ 等待前端编译完成...")
        import webbrowser
        import urllib.request
        
        max_wait = 60  # 最多等待60秒
        for i in range(max_wait):
            if should_exit:
                break
            try:
                urllib.request.urlopen(f"http://localhost:{FRONTEND_PORT}", timeout=2)
                print("🌐 前端就绪，正在打开浏览器...")
                webbrowser.open(f"http://localhost:{FRONTEND_PORT}")
                print("✅ 浏览器已打开\n")
                break
            except:
                time.sleep(1)
        else:
            print("⚠️ 前端启动超时，请手动打开浏览器\n")
        
        # 等待进程或 Ctrl+C
        while not should_exit:
            if backend.poll() is not None:
                print("⚠️ FastAPI 后端已退出")
                break
            if frontend.poll() is not None:
                print("⚠️ Reflex 前端已退出")
                break
            time.sleep(0.5)
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
    
    finally:
        cleanup()


if __name__ == "__main__":
    main()
