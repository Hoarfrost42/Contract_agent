from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class ProgressTracker:
    """In-memory tracker for job logs and results."""

    _instance: Optional["ProgressTracker"] = None
    TTL_SECONDS = 1800  # 30 minutes

    def __new__(cls) -> "ProgressTracker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._jobs: Dict[str, Dict[str, Any]] = {}
        return cls._instance

    def _ensure_job(self, job_id: str) -> None:
        if job_id not in self._jobs:
            self._jobs[job_id] = {
                "logs": [], 
                "status": "pending", 
                "result": None,
                "start_time": time.time()  # Record start time
            }

    def _cleanup_expired(self) -> None:
        """清理已完成且超过 TTL 的任务，防止内存泄漏"""
        now = time.time()
        expired_jobs = [
            job_id for job_id, job_data in self._jobs.items()
            if job_data["status"] == "done" and (now - job_data["start_time"]) > self.TTL_SECONDS
        ]
        for job_id in expired_jobs:
            del self._jobs[job_id]

    def add_log(self, job_id: str, message: str) -> None:
        """Append a log entry for a given job_id and print it to stdout."""
        self._ensure_job(job_id)
        self._jobs[job_id]["logs"].append(message)
        print(f"[{job_id}] {message}")

    def set_result(self, job_id: str, result: Any) -> None:
        """Store the final result and mark job as done."""
        self._ensure_job(job_id)
        self._jobs[job_id]["result"] = result
        self._jobs[job_id]["status"] = "done"
        self._jobs[job_id]["logs"].append("任务已完成。")

    def get_status(self, job_id: str) -> Dict[str, Any]:
        """Return the current status, logs, result, and elapsed time for a job."""
        # 每次查询时清理过期任务
        self._cleanup_expired()
        
        self._ensure_job(job_id)
        entry = self._jobs[job_id]
        
        # Calculate elapsed time
        elapsed_seconds = time.time() - entry["start_time"]
        elapsed_str = f"{elapsed_seconds:.1f}s"
        
        return {
            "logs": list(entry["logs"]),
            "status": entry["status"],
            "result": entry["result"],
            "elapsed": elapsed_str,  # Add elapsed time
        }
