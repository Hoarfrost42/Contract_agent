"""
Reflex 应用状态管理
迁移自 Streamlit session_state，保留 100% 核心逻辑
"""
import reflex as rx
import asyncio
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
import os
import time

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 延迟导入标记 - 不在模块加载时导入重型依赖
CORE_AVAILABLE = True  # 假设可用，在实际调用时再检查

def get_contract_analyzer():
    """延迟导入 ContractAnalyzer"""
    try:
        from src.core.engine import ContractAnalyzer
        return ContractAnalyzer
    except ImportError as e:
        print(f"ContractAnalyzer import error: {e}")
        return None

def get_file_loader():
    """延迟导入 extract_text_from_file"""
    try:
        from src.utils.file_loader import extract_text_from_file
        return extract_text_from_file
    except ImportError as e:
        print(f"File loader import error: {e}")
        return None


class AppState(rx.State):
    """应用全局状态"""
    
    # ==================== 导航状态 ====================
    current_page: str = "home"  # home, report, benchmark
    
    # ==================== 输入状态 ====================
    input_method: str = "upload"  # "upload" or "paste"
    contract_text: str = ""
    uploaded_filename: str = ""
    
    # ==================== 分析状态 ====================
    is_loading: bool = False
    processing_time: float = 0.0
    progress: int = 0
    progress_status: str = "IDLE"
    
    # ==================== 结果状态 ====================
    risk_score: int = 0
    structured_data: List[Dict[str, Any]] = []
    report_md: str = ""
    analysis_complete: bool = False
    
    # ==================== 系统状态 ====================
    system_status: str = "online"  # online, error
    latency: float = 0.0
    notification: str = ""
    
    # ==================== 设置 (隐藏) ====================
    model_selection: str = "deepseek"
    use_cloud_model: bool = False
    enable_deep_reflection: bool = True
    selected_dimensions: List[str] = ["1", "2", "3", "4", "5", "6", "7", "8"]
    
    # ==================== Benchmark 状态 ====================
    ablation_modes: List[int] = [1, 2, 3, 4]
    ablation_data_path: str = "evaluation/llm_benchmark_dataset.json"
    ablation_limit: int = 5
    ablation_source: str = "local"
    ablation_results: Dict[str, Any] = {}
    ablation_running: bool = False
    ablation_chart_paths: List[str] = []  # 图表路径列表
    ablation_combined_chart: str = ""  # 综合图表路径
    
    # 可用数据集列表
    available_datasets: List[str] = [
        "evaluation/llm_benchmark_dataset.json",
    ]
    
    # ==================== 报告页状态 ====================
    expanded_clause_index: int = -1  # -1 表示全部折叠
    report_view_mode: str = "summary"  # "summary" 或 "cards"
    word_report_path: str = ""  # Word 报告路径
    word_export_loading: bool = False  # 导出中状态
    
    def set_report_view_mode(self, mode: str):
        """切换报告展示模式"""
        self.report_view_mode = mode
    
    async def export_word_report(self):
        """导出 Word 报告"""
        if not self.structured_data:
            self.notification = "⚠️ 暂无报告数据可导出"
            return
        
        self.word_export_loading = True
        self.notification = "📄 正在生成 Word 报告..."
        yield
        
        try:
            from src.utils.word_exporter import generate_word_report
            from datetime import datetime
            
            # 生成 Word 报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = project_root / "temp_reports"
            
            report_path = generate_word_report(
                structured_data=self.structured_data,
                report_md=self.report_md,
                risk_score=self.risk_score,
                output_dir=output_dir,
                filename=f"合同风险分析报告_{timestamp}.docx"
            )
            
            self.word_report_path = report_path
            self.notification = "✅ Word 报告生成成功，可以下载"
            
        except Exception as e:
            self.notification = f"❌ 导出失败: {str(e)}"
            
        finally:
            self.word_export_loading = False
    
    # ==================== 计算属性 ====================
    @rx.var
    def processing_time_formatted(self) -> str:
        """格式化耗时显示"""
        return f"{self.processing_time:.1f}"
    
    @rx.var
    def high_risk_count(self) -> int:
        """高风险条款数量"""
        return sum(1 for item in self.structured_data if item.get("risk_level") == "高")
    
    @rx.var
    def medium_risk_count(self) -> int:
        """中风险条款数量"""
        return sum(1 for item in self.structured_data if item.get("risk_level") == "中")
    
    @rx.var
    def low_risk_count(self) -> int:
        """低风险条款数量"""
        return sum(1 for item in self.structured_data if item.get("risk_level") == "低")
    
    # ==================== 导航方法 ====================
    def navigate_to(self, page: str):
        """导航到指定页面"""
        self.current_page = page
        
    def go_home(self):
        self.current_page = "home"
        self.analysis_complete = False
        
    def go_report(self):
        self.current_page = "report"
        
    def go_benchmark(self):
        self.current_page = "benchmark"
    
    # ==================== 输入方法 ====================
    def set_input_method(self, value: str):
        self.input_method = value
        
    def set_contract_text(self, value: str):
        self.contract_text = value
    
    def set_enable_deep_reflection(self, value: bool):
        """切换深度反思模式"""
        self.enable_deep_reflection = value
    
    # ==================== 文件上传处理 ====================
    async def handle_upload(self, files: List[rx.UploadFile]):
        """处理文件上传 - 保留原有解析逻辑"""
        self.is_loading = True
        self.notification = ""
        
        for file in files:
            try:
                upload_data = await file.read()
                filename = file.filename
                self.uploaded_filename = filename
                
                # 保存到临时目录
                temp_dir = project_root / "temp_uploads"
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / filename
                
                with open(temp_path, "wb") as f:
                    f.write(upload_data)
                
                # 使用原有的文件解析逻辑 (延迟导入)
                extract_fn = get_file_loader()
                if extract_fn:
                    # 调用 extract_text_from_file
                    with open(temp_path, "rb") as f:
                        self.contract_text = extract_fn(f)
                    self.notification = f"✅ 已加载: {filename} ({len(self.contract_text)} 字符)"
                else:
                    # 降级处理
                    if filename.endswith(".txt"):
                        self.contract_text = upload_data.decode("utf-8")
                    else:
                        self.notification = "⚠️ 核心模块不可用，仅支持 TXT 文件"
                        
            except Exception as e:
                self.notification = f"❌ 上传失败: {str(e)}"
        
        self.is_loading = False
    
    # ==================== 分析方法 (通过 FastAPI 后端) ====================
    API_BASE_URL = "http://127.0.0.1:8002"
    
    async def run_analysis(self):
        """通过 FastAPI 后端执行合同分析"""
        import requests
        
        if not self.contract_text:
            self.notification = "⚠️ 请先上传文件或粘贴文本"
            return
            
        self.is_loading = True
        self.progress = 0
        self.progress_status = "INITIALIZING"
        self.notification = "正在提交分析任务..."
        self.structured_data = []
        self.report_md = ""
        
        start_time = time.time()
        yield
        
        try:
            # 1. 提交任务到 FastAPI 后端
            payload = {
                "text": self.contract_text,
                "llm_source": "cloud" if self.use_cloud_model else "local",
                "deep_reflection": self.enable_deep_reflection
            }
            
            resp = requests.post(f"{self.API_BASE_URL}/submit", json=payload, timeout=10)
            resp.raise_for_status()
            job_data = resp.json()
            job_id = job_data.get("job_id")
            
            if not job_id:
                self.notification = "❌ 任务提交失败"
                self.is_loading = False
                return
            
            self.notification = f"✅ 任务已提交 (ID: {job_id[:8]}...)"
            self.progress = 10
            yield
            
            # 2. 轮询进度
            max_wait = 300  # 最长等待 5 分钟
            poll_interval = 2  # 每 2 秒轮询一次
            elapsed = 0
            
            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                
                try:
                    status_resp = requests.get(f"{self.API_BASE_URL}/progress/{job_id}", timeout=5)
                    status_data = status_resp.json()
                    status = status_data.get("status", "unknown")
                    logs = status_data.get("logs", [])
                    
                    # 更新进度
                    self.progress_status = status.upper()
                    self.progress = min(10 + int(elapsed / max_wait * 80), 90)
                    if logs:
                        self.notification = logs[-1]
                    yield
                    
                    # 检查是否完成
                    if status == "done":
                        result = status_data.get("result", {})
                        self.structured_data = result.get("results", [])
                        self.report_md = result.get("report", "")
                        self.risk_score = result.get("risk_score", 0)
                        self.analysis_complete = True
                        
                        self.processing_time = time.time() - start_time
                        self.progress = 100
                        self.progress_status = "DONE"
                        self.notification = f"✅ 分析完成 (耗时 {self.processing_time:.1f}s)"
                        self.is_loading = False
                        yield rx.redirect("/report")
                        return
                    
                    elif status == "error":
                        self.notification = f"❌ 分析失败: {status_data.get('error', '未知错误')}"
                        self.is_loading = False
                        return
                        
                except requests.RequestException as e:
                    self.notification = f"⚠️ 进度查询失败: {e}"
                    yield
            
            # 超时
            self.notification = "❌ 分析超时，请重试"
            self.is_loading = False
                
        except requests.RequestException as e:
            self.notification = f"❌ 后端连接失败: {e}. 请确保 FastAPI 后端正在运行 (python -m src.api.main)"
            self.is_loading = False
            
        except Exception as e:
            self.notification = f"❌ 分析失败: {str(e)}"
            self.progress_status = "ERROR"
            self.system_status = "error"
            self.is_loading = False
            
        finally:
            self.is_loading = False
    
    # ==================== 报告页方法 ====================
    def toggle_clause(self, index: int):
        """展开/折叠条款详情"""
        if self.expanded_clause_index == index:
            self.expanded_clause_index = -1
        else:
            self.expanded_clause_index = index
    
    # ==================== Benchmark 方法 ====================
    def toggle_ablation_mode(self, mode: int):
        """切换消融模式选中状态"""
        if mode in self.ablation_modes:
            self.ablation_modes.remove(mode)
        else:
            self.ablation_modes.append(mode)
            self.ablation_modes.sort()
    
    def set_ablation_path(self, value: str):
        self.ablation_data_path = value
        
    def set_ablation_limit(self, value: str):
        # rx.input 传递字符串，在此处转换
        try:
            self.ablation_limit = int(value) if value else 5
        except ValueError:
            self.ablation_limit = 5
        
    def set_ablation_source(self, value: str):
        self.ablation_source = value
    
    async def run_ablation(self):
        """运行消融实验"""
        if not self.ablation_modes:
            self.notification = "⚠️ 请至少选择一个评测模式"
            return
            
        self.ablation_running = True
        self.notification = "🔬 消融实验运行中..."
        
        yield
        
        try:
            # 导入消融模块
            from evaluation.ablation_benchmark import run_full_ablation_study
            
            # 转换为绝对路径（相对于项目根目录）
            data_path = str(project_root / self.ablation_data_path)
            
            results = await run_full_ablation_study(
                data_path=data_path,
                modes=self.ablation_modes,
                limit=self.ablation_limit,
                source=self.ablation_source
            )
            
            self.ablation_results = results
            
            # 提取图表路径
            if "chart_paths" in results:
                self.ablation_chart_paths = results.get("chart_paths", [])
            if "combined_chart" in results:
                self.ablation_combined_chart = results.get("combined_chart", "")
                
            self.notification = "✅ 消融实验完成"
            
        except Exception as e:
            self.notification = f"❌ 实验失败: {str(e)}"
            
        finally:
            self.ablation_running = False
