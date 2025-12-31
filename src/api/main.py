from __future__ import annotations

import asyncio
import logging
import warnings
from uuid import uuid4

# Suppress pkg_resources deprecation warning from jieba
warnings.filterwarnings("ignore", category=UserWarning, module='jieba')

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import AnalyzeRequest
from src.core.engine import ContractAnalyzer
from src.utils.progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

app = FastAPI(title="Contract Risk Analyzer", version="1.0.0")

import os

# CORS 配置：从环境变量读取允许的域名，默认仅允许 localhost
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:9000,http://localhost:8501").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局实例
TRACKER = ProgressTracker()

async def _run_job(job_id: str, text: str, llm_source: str, deep_reflection: bool = False):
    """后台运行分析任务的包装函数"""
    analyzer = ContractAnalyzer()
    # 注入全局追踪器，以便通过 API 查询状态
    analyzer.tracker = TRACKER 
    await analyzer.analyze(job_id, text, llm_source, deep_reflection)

@app.post("/submit")
async def submit_job(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """提交分析任务接口"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="合同文本不能为空。")

    job_id = str(uuid4())
    mode_desc = "深度反思" if request.deep_reflection else "标准"
    TRACKER.add_log(job_id, f"任务已提交 (模型源: {request.llm_source}, 模式: {mode_desc})，正在初始化分析引擎...")
    # 添加后台任务
    background_tasks.add_task(_run_job, job_id, request.text, request.llm_source, request.deep_reflection)
    return {"job_id": job_id}


@app.get("/progress/{job_id}")
async def get_progress(job_id: str) -> dict:
    """查询任务进度接口"""
    return TRACKER.get_status(job_id)


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8002, reload=True)
