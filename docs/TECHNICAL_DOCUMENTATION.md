# 合同风险识别系统 - 技术文档

## 一、系统概述

本系统是一个基于大语言模型 (LLM) 的智能合同风险识别平台，结合规则引擎与深度学习技术，实现合同条款的自动化风险分析与评估。

### 核心能力

| 能力 | 描述 |
|------|------|
| **条款级风险识别** | 自动切分合同条款，逐条进行风险分析 |
| **规则+LLM 混合架构** | 规则引擎预匹配 + LLM 深度分析 |
| **自反思机制** | 对高风险结果进行二次审查，降低误判 |
| **消融实验评测** | 支持多模式对比评测，量化各组件贡献 |

---

## 二、技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Reflex 前端 (3000)                       │
│                   (Single-Page Application)                  │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP API
┌───────────────────────────▼─────────────────────────────────┐
│                    FastAPI 后端 (8002)                       │
│              /submit  /progress/{job_id}                     │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   ContractAnalyzer 核心引擎                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ 文本预处理   │ →  │  规则引擎   │ →  │  LLM 分析客户端  │  │
│  │ (条款切分)   │    │ (关键词匹配) │    │  (Ollama API)   │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP (requests)
┌───────────────────────────▼─────────────────────────────────┐
│                    Ollama LLM 服务 (11434)                   │
│                  qwen3:4b-instruct / deepseek                │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、技术栈详解

### 3.1 前端技术

| 技术 | 版本 | 用途 |
|------|------|------|
| **Reflex** | latest | 全栈 Python Web 框架，基于 React |
| Python | 3.12 | 前后端统一语言 |

**核心文件：**
- `reflex_web/reflex_web/reflex_web.py` - 主应用入口与页面路由
- `reflex_web/reflex_web/state.py` - 全局状态管理
- `reflex_web/reflex_web/components/` - UI 组件（report.py, benchmark.py, sidebar.py）

### 3.2 后端 API

| 技术 | 版本 | 用途 |
|------|------|------|
| **FastAPI** | ≥0.115.0 | RESTful API 框架 |
| **Uvicorn** | ≥0.32.0 | ASGI 服务器 |
| **Pydantic** | ≥2.0.0 | 数据验证与序列化 |

**API 端点：**
| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/submit` | 提交合同分析任务 |
| GET | `/progress/{job_id}` | 查询任务进度与结果 |

### 3.3 核心引擎

| 模块 | 文件 | 功能 |
|------|------|------|
| **ContractAnalyzer** | `src/core/engine.py` | 主分析工作流编排 |
| **RuleEngine** | `src/core/rule_engine.py` | 规则库匹配与预判 |
| **LLMClient** | `src/core/llm.py` | LLM API 调用封装 |
| **Preprocessor** | `src/core/preprocessor.py` | 文本预处理与清洗 |

**分析流程：**
1. 文本预处理：切分合同为独立条款
2. 规则引擎预匹配：基于关键词定位高风险条款
3. LLM 深度分析：对预匹配结果进行语义分析
4. 自反思审查（可选）：对高风险判定进行二次校验
5. 报告生成：汇总分析结果生成 Markdown 报告

### 3.4 LLM 服务

| 技术 | 说明 |
|------|------|
| **Ollama** | 本地 LLM 运行时 |
| **qwen3:4b-instruct** | 默认模型（本地） |
| **deepseek-chat** | 云端模型备选 |

**调用方式：** 使用 `requests` 库直接调用 Ollama REST API（`/api/chat`, `/api/generate`），避免 `httpx` 兼容性问题。

**配置文件：** `configs/config.yaml`
```yaml
llm_config:
  provider: "ollama"
  base_url: "http://localhost:11434"
  model_name: "qwen3:4b-instruct"
  temperature: 0
```

### 3.5 规则引擎

| 组件 | 说明 |
|------|------|
| **risk_rules.json** | 107 条专家规则库 |
| **RuleEngine** | 关键词匹配 + 置信度计算 |

**规则结构：**
```json
{
  "risk_id": "GENERAL_001",
  "contract_type": "通用条款",
  "risk_name": "异地高成本管辖",
  "keywords": ["甲方所在地法院", "指定管辖"],
  "risk_type": "commercial",
  "analysis_logic": "...",
  "laws": "《民事诉讼法》第三十五条"
}
```

---

## 四、评测系统

### 4.1 消融实验框架

| 模式 | 描述 | 组件 |
|------|------|------|
| **Mode 1** | 纯 LLM (Raw) | 无 Prompt 模板，直接输入 |
| **Mode 2** | 基础 Prompt | 格式化 Prompt，无规则引擎 |
| **Mode 3** | 当前工作流 | Prompt + 规则引擎 |
| **Mode 4** | 优化工作流 | CoT Prompt + 规则引擎 |

### 4.2 评测指标

| 指标 | 类型 | 描述 |
|------|------|------|
| `accuracy` | 核心 | 风险等级预测正确率 |
| `precision` | 核心 | TP / (TP + FP) |
| `recall` | 核心 | TP / (TP + FN) |
| `f1` | 核心 | Precision 与 Recall 调和平均 |
| `parse_rate` | 实用 | LLM 输出格式解析成功率 |
| `hallucination_rate` | 重要 | 证据不在原文中的比例 |
| `risk_id_accuracy` | 新增 | 正例样本风险识别准确率 |
| `avg_latency_sec` | 新增 | 平均响应时间（秒） |

### 4.3 评测数据集

| 数据集 | 样本数 | 来源 |
|--------|--------|------|
| `llm_benchmark_dataset.jsonl` | 321 | LLM 生成（正例+负例+边界） |
| `generated_test_cases.jsonl` | 428 | 规则库自动生成 |
| `golden_dataset.jsonl` | 8 | 人工标注样本 |

---

## 五、项目结构

```
f:\Agent_back\11.29\
├── src/                          # 核心源码
│   ├── api/                      # FastAPI 后端
│   │   ├── main.py              # API 入口
│   │   └── schemas.py           # 请求/响应模型
│   ├── core/                     # 核心引擎
│   │   ├── engine.py            # ContractAnalyzer
│   │   ├── llm.py               # LLM 客户端
│   │   ├── rule_engine.py       # 规则引擎
│   │   ├── risk_rules.json      # 风险规则库 (107条)
│   │   └── types.py             # 数据类型定义
│   └── utils/                    # 工具模块
│       ├── parser.py            # 合同切分
│       ├── file_loader.py       # 文件解析
│       └── progress_tracker.py  # 进度追踪
├── reflex_web/                   # Reflex 前端
│   └── reflex_web/
│       ├── reflex_web.py        # 主应用
│       ├── state.py             # 状态管理
│       └── components/          # UI 组件
├── evaluation/                   # 评测模块
│   ├── ablation_benchmark.py    # 消融实验脚本
│   ├── generate_test_cases.py   # 测试用例生成
│   └── llm_benchmark_dataset.jsonl  # 评测数据集
├── configs/
│   ├── config.yaml              # 主配置文件
│   └── risk_standards.yaml      # 风险评估标准
├── start_reflex.py              # 启动脚本
└── requirements.txt             # Python 依赖
```

---

## 六、依赖清单

### 核心依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| langchain-core | ≥0.3.0 | LangChain 核心 |
| langchain-community | ≥0.3.0 | 社区集成 |
| langgraph | ≥0.2.0 | 工作流编排 |
| fastapi | ≥0.115.0 | Web API |
| uvicorn | ≥0.32.0 | ASGI 服务器 |
| pydantic | ≥2.0.0 | 数据验证 |
| requests | ≥2.32.0 | HTTP 客户端 |

### 文件处理

| 包名 | 版本 | 用途 |
|------|------|------|
| python-docx | ≥1.1.0 | Word 文档解析 |
| pdfplumber | ≥0.11.0 | PDF 文档解析 |

---

## 七、运行指南

### 7.1 环境准备

```bash
# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows

# 2. 安装依赖
pip install -r requirements.txt
pip install reflex

# 3. 启动 Ollama
ollama serve
ollama pull qwen3:4b-instruct
```

### 7.2 启动服务

```bash
# 一键启动（推荐）
python start_reflex.py

# 分别启动
python -m src.api.main  # FastAPI 后端 (8002)
cd reflex_web && reflex run  # Reflex 前端 (3000)
```

### 7.3 运行评测

```bash
# 单模式评测
python evaluation/ablation_benchmark.py --mode 2 --limit 10

# 完整消融实验
python evaluation/ablation_benchmark.py
```

---

## 八、已知问题与解决方案

### 8.1 httpx 与 Ollama 兼容性问题

**问题：** `langchain_ollama` 和 `ollama` 库使用 `httpx` 发送请求，与 Ollama 0.13.5 存在兼容性问题，返回 502 错误。

**解决方案：** 使用 `requests` 库直接调用 Ollama REST API。

```python
# ❌ 旧方式（不兼容）
from langchain_ollama import ChatOllama
llm = ChatOllama(model="qwen3:4b-instruct")

# ✅ 新方式（兼容）
import requests
response = requests.post(
    "http://localhost:11434/api/chat",
    json={"model": "qwen3:4b-instruct", "messages": [...], "stream": False}
)
```

---

*文档更新日期：2025-12-29*
