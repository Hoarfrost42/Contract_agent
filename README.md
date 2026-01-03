# Contract Risk Agent (合同风险监测智能体)
基于大语言模型的合同风险监测智能体
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Reflex](https://img.shields.io/badge/Reflex-Latest-000000?logo=reflex&logoColor=white)](https://reflex.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?logo=langchain)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-0.13+-000000?logo=ollama&logoColor=white)](https://ollama.ai)

---

##  项目背景与概述

基于垂直领域大模型种类缺乏，而法律行业在合同审查方面需要大量的人力、时间成本的背景，同时合同领域对数据隐私性敏感，难以通过调用云端的大模型高频率大范围地使用。本项目使用了Qwen3:4b-instruct这一便于本地部署的小模型，旨在通过足够的先验知识与Prompt约束引导，能在一定程度上提升合同审查的效率。

系统采用 **"Rule-Guided + LLM-Reasoning"** (规则引导 + 大模型推理) 的混合架构：
1. **规则引擎**：这一部分专家知识蒸馏自GPT-5，Gemini 3 Pro等目前主流高性能大模型，经过人工验证和模型之间的相互验证。用于提供固定程式下最为常见的合同风险的辨别。
2. **大语言模型推理**：Qwen3：4b-instruct作为小参数模型，易于本地部署，同时也有相对充分的预训练知识，对于已经提供好先验知识和推理框架的合同风险分析，能表现出比较优良的性能。
> ⚠️ **免责声明**：本项目为学术研究与技术验证性质，仅供学习交流，不构成任何法律建议。在签署重要合同时，请务必咨询专业律师。

---

## 系统架构
为了提供更现代化的交互体验和更稳定的系统架构，本项目在当前版本使用了 **Reflex** + **FastAPI**的架构。

### 主要优点
*   **交互体验**
    *   **双面板即时交互**: 首页采用 Split-View 设计，左侧操作与右侧预览实时同步，无需频繁跳转。分析完成后即时跳转报告页面。
    *   **无缝工作流**: 文件拖拽后自动上传解析。

*   **系统架构解耦**
    *   **前端**: 使用 **Reflex** (基于 React) 构建高性能前端，状态管理更加可控。
    *   **后端**: 独立封装 **FastAPI** 服务层，负责处理耗时的 LLM 分析与检索任务，通过异步接口与前端通信。

*   **推理表现**
    *   **三级风险评估体系**: 支持 🔴高/🟠中/🟢低 三级风险分类，基于合法性、经济性、确定性三维评估。分级明确，减少因为风险划分不明显导致的歧义。
    *   **深度反思模式**: 对高/中风险结果触发大模型的二次审查，产生「维持/调级/存疑」三种结论，保证最终推理效果趋于稳定，多次推理一致性较好。调级逻辑支持双向调整。
    *   **Top-K 规则匹配**: 支持召回多个候选规则（可配置 top_k），为 LLM 提供更丰富的专家知识上下文。
    *   **透明化推理**: 在终端实时输出规则匹配链路（匹配规则 -> 置信度 -> 法律依据），让 AI 的决策过程"可解释"。

---

## 核心功能

### 1. 多格式输入
*   **多格式支持**: 集成了 `pdfplumber`, `python-docx`, `chardet` 等库，支持 PDF、Word、TXT 格式合同的精准文本提取。
*   **智能清洗**: 自动去除页眉页脚、乱码干扰，保留文档核心结构。

### 2. 结构化条款切分
*   **正则切分器**: 基于法律文档排版规范，利用正则表达式精确识别"第一条"、"1."、"1.1"等层级结构。
*   **中文数字标准化**: 内置算法将"第一百二十三条"标准化为数字索引，便于后续处理。

### 3. 混合规则检索引擎 (Hybrid Retrieval + Rerank)
*   **双路召回**:
    *   **语义检索**: 使用 `BGE-small-zh-v1.5` 生成高维语义向量，捕获"语义相似"的风险规则。
    *   **关键词匹配**: 使用 `BM25Okapi` + `jieba` 分词，捕获"关键词精确匹配"的风险规则。
*   **加权融合**: 采用 `alpha * Dense + (1-alpha) * Sparse` 加权求和算法融合两路结果（默认 alpha=0.5），避免语义匹配产生歧义或者纯关键词匹配过于死板。
*   **Reranker 重排序**: 使用 `BAAI/bge-reranker-v2-m3` 对初步检索结果进行二次精排，过滤低相关性候选（默认阈值 0.3），显著提升参考信息的质量。
*   **Top-K 多规则召回**: 支持返回多个超过置信度阈值的候选规则（默认 top_k=3），提供更丰富的上下文。
*   **空上下文安全兜底**: 当 Reranker 过滤后参考信息为空时，系统向 LLM 发送明确信号，引导模型基于通用法律常识判断，避免幻觉。

### 4. 法律与专家知识库
*   **风险规则库**: 内置 120+ 条经过法律专家标注的常见商业合同风险点（如：霸王条款、不合理违约金、管辖权陷阱），主要涵盖劳动、买卖、租赁三大领域，并且辅以通用条款风险保证全面。
*   **法条数据库**: 内置 SQLite 数据库，存储《民法典》、《劳动合同法》等核心法条，支持精确引用。

### 5. 智能预处理与风险检测
*   **开场白过滤**: 自动识别合同开头的程序性内容（如"甲乙双方协商一致签订本合同"），避免误报。
*   **薪资结构失衡检测**: 自动解析薪资条款，当底薪占比低于 40% 时发出警告（防止"高绩效低底薪"陷阱）。

### 6. LLM 深度推理与反思
*   **三级风险定义标准**:
    *   **🔴 高风险**: 违法/无效/双倍赔偿/核心权益剥夺（条款显式对抗法律强制性规定）。
    *   **🟠 中风险**: 模糊/歧义/举证困难/隐形损失（条款沉默或模糊导致解释不确定）。
    *   **🟢 低风险**: 法律自动补位/程序性瑕疵（法律有兜底条款，无实质损失）。
*   **提示词工程**: 将召回的规则、法条、条款上下文动态注入 Prompt，约束 LLM 的生成范围。
*   **深度反思**: 对高/中风险触发二次审查，支持「维持/调级/存疑」三种结论。

### 7. 风险报告展示
*   **结构化输出**: 生成 Markdown 格式的详细报告，包含风险等级、风险点摘要、法条依据、修改建议。
*   **可视化**: 提供风险概览仪表盘，支持按高/中/低风险等级筛选查看。
*   **Word 导出**: 可一键导出 Word 格式报告。

---

## 技术栈

| 模块 | 技术选型 | 说明 |
| :--- | :--- | :--- |
| **前端框架** | **Reflex** | 全栈 Python Web 框架，编译为 React SPA，提供极致交互体验 |
| **后端服务** | **FastAPI** | 高性能异步 API 框架，承载核心业务逻辑 |
| **任务队列** | **BackgroundTasks** | FastAPI 原生后台任务，处理长耗时分析 |
| **大模型框架** | **LangChain** | 负责 Prompt 编排、模型调用接口封装 |
| **本地推理** | **Ollama** | 运行 qwen3:4b-instruct / DeepSeek 等开源大模型 |
| **LLM 调用** | **requests** | 直接调用 Ollama REST API，避免 httpx 兼容性问题 |
| **向量模型** | **BGE-small-zh-v1.5** | 中文语义向量表征 (Sentence-Transformers) |
| **重排序模型** | **BGE-Reranker-v2-m3** | 对检索结果进行二次精排，过滤低相关性候选 |
| **全文检索** | **Rank-BM25** | 传统的词袋模型检索，保证关键词命中率 |
| **数据存储** | **SQLite** | 轻量级关系型数据库，存储法条与历史记录 |

---

## 消融实验评测系统

系统内置了完整的消融实验框架，用于量化各组件对最终效果的贡献。主要用于项目继续改进的评估。

### 评测模式
以下四种模式均规定了输出格式便于解析。无Prompt指的是没有角色引导、先验知识等的注入。

| 模式 | 描述 | 组件 |
|------|------|------|
| **Mode 1** | 纯 LLM (Raw) | 无 Prompt 模板，直接输入 |
| **Mode 2** | 基础 Prompt | 格式化 Prompt，无规则引擎 |
| **Mode 3** | 当前工作流 | Prompt + 规则引擎 |
| **Mode 4** | 优化工作流 | CoT Prompt + 规则引擎 |

### 评测指标

| 指标 | 描述 |
|------|------|
| `accuracy` | 风险等级预测正确率 |
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) |
| `f1` | Precision 与 Recall 调和平均 |
| `hallucination_rate` | 证据不在原文中的比例 |
| `risk_id_accuracy` | 正例样本风险识别准确率 |
| `avg_latency_sec` | 平均响应时间（秒） |

### 运行评测

```bash
# 单模式评测
python evaluation/ablation_benchmark.py --mode 2 --limit 10

# 完整消融实验
python evaluation/ablation_benchmark.py --limit 20
```

> 详细技术文档请见：[TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md)

以上操作均可通过start_reflex.py脚本启动后在Web端进行。

## 快速开始-How to start

### 1. 环境准备
确保已安装 Python 3.10+ 和 Git。

```bash
# 克隆项目代码
git clone https://github.com/your-repo/contract-ai-agent.git
cd contract-ai-agent

# 创建并激活虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 模型准备
#### A. LLM 模型 (Ollama)
下载并安装 [Ollama](https://ollama.ai/)，然后拉取模型：
```bash
# 默认模型（推荐）
ollama pull qwen3:4b-instruct

# 或者使用大参数模型
ollama pull qwen2.5:14b
```

#### B. Embedding 模型
下载 `BAAI/bge-small-zh-v1.5` 模型到本地，并在 `configs/config.yaml` 中配置路径：
```yaml
embedding_config:
  model_path: "/path/to/your/bge-small-zh-v1.5"
```
*(注：也可以简单使用 HuggingFace 自动下载，无需手动配置路径)*

### 4. 启动系统
使用项目根目录下的启动脚本，一键拉起后端 API 和前端界面（启动完成后可自动跳转）：

```bash
python start_reflex.py
```

*   **访问前端**: 打开浏览器访问 `http://localhost:3000`
*   **查看文档**: 后端 API 文档位于 `http://localhost:8002/docs`

---

## 核心配置说明 (`configs/config.yaml`)

| 配置项 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `system_config.max_concurrency` | `5` | 并发分析的条款数量，显存越大可调越大 |
| `llm_config.model_name` | `qwen3:4b-instruct` | 使用的 LLM 模型名称，需与 Ollama 一致 |
| `llm_config.temperature` | `0` | 模型生成的随机度，法律场景建议低值以保准确 |
| `hybrid_search_config.threshold` | `0.75` | 规则匹配的最低置信度阈值 |
| `hybrid_search_config.alpha` | `0.65` | 混合检索权重系数 (0.65 代表 65% 语义 + 65% 关键词) |

---

## 项目结构

```
Contract-AI-Agent/
├── configs/                 # 配置文件 (模型参数、规则阈值)
│   ├── config.yaml          # 主配置
│   └── risk_standards.yaml  # 风险评估标准
├── data/                    # 数据资源
│   └── databases/           # 向量数据库与法条库
├── docs/                    # 文档目录
│   ├── weekly_reports/      # 周报
│   ├── reports/             # Word 报告
│   └── *.md                 # 技术文档
├── evaluation/              # 评测模块
│   ├── datasets/sources/    # 评测源数据 (GENERAL/LABOR/LEASE/SALES)
│   ├── results/             # 评测结果输出
│   ├── ablation_benchmark.py  # 消融实验主脚本
│   └── run_benchmark.py     # 评测运行器
├── reflex_web/              # Reflex 前端
│   ├── reflex_web/
│   │   ├── components/      # UI 组件 (卡片、侧边栏、报告视图)
│   │   ├── state.py         # 前端状态管理 (AppState)
│   │   ├── styles.py        # 样式定义 (Glassmorphism)
│   │   └── reflex_web.py    # 页面入口 (Index, Report)
│   └── assets/              # 静态资源图片
├── scripts/                 # 工具脚本
│   ├── import_processed_laws.py  # 法条导入
│   ├── process_law_text.py       # 法条处理
│   └── test_llm.py               # LLM 测试
├── src/                     # 核心源码
│   ├── api/                 # FastAPI 后端接口
│   ├── core/                # 核心业务逻辑
│   │   ├── engine.py        # 分析主引擎
│   │   ├── rule_engine.py   # 规则匹配引擎
│   │   ├── llm.py           # LLM 调用封装
│   │   ├── hybrid_searcher.py # 混合检索器
│   │   ├── reference_retriever.py # 统一检索器 (Top-K + Rerank)
│   │   ├── reranker.py      # BGE-Reranker 重排序模块
│   │   ├── prompts.py       # Prompt 模板集中管理
│   │   ├── output_parser.py # LLM 输出解析器
│   │   ├── contract_classifier.py # 合同类型分类器
│   │   └── ollama_client.py # Ollama API 客户端
│   ├── database/            # 数据库操作
│   └── utils/               # 通用工具函数
├── .gitignore
├── README.md
├── requirements.txt
├── start_reflex.py          # 启动脚本
└── start_services.py        # 服务启动脚本
```
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Reflex](https://img.shields.io/badge/Reflex-Latest-000000?logo=reflex&logoColor=white)](https://reflex.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?logo=langchain)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-0.13+-000000?logo=ollama&logoColor=white)](https://ollama.ai)

---

## � 项目背景与概述

基于垂直领域大模型种类缺乏，而法律行业在合同审查方面需要大量的人力、时间成本的背景，同时合同领域对数据隐私性敏感，难以通过调用云端的大模型高频率大范围地使用。本项目使用了Qwen3:4b-instruct这一便于本地部署的小模型，旨在通过足够的先验知识与Prompt约束引导，能在一定程度上提升合同审查的效率。

系统采用 **"Rule-Guided + LLM-Reasoning"** (规则引导 + 大模型推理) 的混合架构：
1. **规则引擎**：这一部分专家知识蒸馏自GPT-5，Gemini 3 Pro等目前主流高性能大模型，经过人工验证和模型之间的相互验证。用于提供固定程式下最为常见的合同风险的辨别。
2. **大语言模型推理**：Qwen3：4b-instruct作为小参数模型，易于本地部署，同时也有相对充分的预训练知识，对于已经提供好先验知识和推理框架的合同风险分析，能表现出比较优良的性能。
> ⚠️ **免责声明**：本项目为学术研究与技术验证性质，仅供学习交流，不构成任何法律建议。在签署重要合同时，请务必咨询专业律师。

---

## 系统架构
为了提供更现代化的交互体验和更稳定的系统架构，本项目在当前版本使用了 **Reflex** + **FastAPI**的架构。

### 主要优点
*   **交互体验**
    *   **双面板即时交互**: 首页采用 Split-View 设计，左侧操作与右侧预览实时同步，无需频繁跳转。分析完成后即时跳转报告页面。
    *   **无缝工作流**: 文件拖拽后自动上传解析。

*   **系统架构解耦**
    *   **前端**: 使用 **Reflex** (基于 React) 构建高性能前端，状态管理更加可控。
    *   **后端**: 独立封装 **FastAPI** 服务层，负责处理耗时的 LLM 分析与检索任务，通过异步接口与前端通信。

*   **推理表现**
    *   **三级风险评估体系**: 支持 🔴高/🟠中/🟢低 三级风险分类，基于合法性、经济性、确定性三维评估。分级明确，减少因为风险划分不明显导致的歧义。
    *   **深度反思模式**: 对高/中风险结果触发大模型的二次审查，产生「维持/调级/存疑」三种结论，保证最终推理效果趋于稳定，多次推理一致性较好。调级逻辑支持双向调整。
    *   **Top-K 规则匹配**: 支持召回多个候选规则（可配置 top_k），为 LLM 提供更丰富的专家知识上下文。
    *   **透明化推理**: 在终端实时输出规则匹配链路（匹配规则 -> 置信度 -> 法律依据），让 AI 的决策过程"可解释"。

---

## 核心功能

### 1. 多格式输入
*   **多格式支持**: 集成了 `pdfplumber`, `python-docx`, `chardet` 等库，支持 PDF、Word、TXT 格式合同的精准文本提取。
*   **智能清洗**: 自动去除页眉页脚、乱码干扰，保留文档核心结构。

### 2. 结构化条款切分
*   **正则切分器**: 基于法律文档排版规范，利用正则表达式精确识别"第一条"、"1."、"1.1"等层级结构。
*   **中文数字标准化**: 内置算法将"第一百二十三条"标准化为数字索引，便于后续处理。

<<<<<<< HEAD
### 3. 混合规则检索引擎 (Hybrid Retrieval + Rerank)
*   **双路召回**:
    *   **语义检索**: 使用 `BGE-small-zh-v1.5` 生成高维语义向量，捕获"语义相似"的风险规则。
    *   **关键词匹配**: 使用 `BM25Okapi` + `jieba` 分词，捕获"关键词精确匹配"的风险规则。
*   **加权融合**: 采用 `alpha * Dense + (1-alpha) * Sparse` 加权求和算法融合两路结果（默认 alpha=0.5），避免语义匹配产生歧义或者纯关键词匹配过于死板。
*   **Reranker 重排序**: 使用 `BAAI/bge-reranker-v2-m3` 对初步检索结果进行二次精排，过滤低相关性候选（默认阈值 0.3），显著提升参考信息的质量。
=======
### 3. 混合规则检索引擎 (Hybrid Retrieval)
*   **双路召回**:
    *   **语义检索**: 使用 `BGE-small-zh-v1.5` 生成高维语义向量，捕获"语义相似"的风险规则。
    *   **关键词匹配**: 使用 `BM25Okapi` + `jieba` 分词，捕获"关键词精确匹配"的风险规则。
*   **加权融合**: 采用 `α × Dense + (1-α) × Sparse` 加权求和算法融合两路结果（默认 α=0.5），避免语义匹配产生歧义或者纯关键词匹配过于死板。
>>>>>>> a24aa9bc36aad0e9578e26f7911cfdc68d1975ee
*   **Top-K 多规则召回**: 支持返回多个超过置信度阈值的候选规则（默认 top_k=3），提供更丰富的上下文。
*   **空上下文安全兜底**: 当 Reranker 过滤后参考信息为空时，系统向 LLM 发送明确信号，引导模型基于通用法律常识判断，避免幻觉。

### 4. 法律与专家知识库
*   **风险规则库**: 内置 120+ 条经过法律专家标注的常见商业合同风险点（如：霸王条款、不合理违约金、管辖权陷阱），主要涵盖劳动、买卖、租赁三大领域，并且辅以通用条款风险保证全面。
*   **法条数据库**: 内置 SQLite 数据库，存储《民法典》、《劳动合同法》等核心法条，支持精确引用。

### 5. 智能预处理与风险检测
*   **开场白过滤**: 自动识别合同开头的程序性内容（如"甲乙双方协商一致签订本合同"），避免误报。
*   **薪资结构失衡检测**: 自动解析薪资条款，当底薪占比低于 40% 时发出警告（防止"高绩效低底薪"陷阱）。

### 6. LLM 深度推理与反思
*   **三级风险定义标准**:
    *   **🔴 高风险**: 违法/无效/双倍赔偿/核心权益剥夺（条款显式对抗法律强制性规定）。
    *   **🟠 中风险**: 模糊/歧义/举证困难/隐形损失（条款沉默或模糊导致解释不确定）。
    *   **🟢 低风险**: 法律自动补位/程序性瑕疵（法律有兜底条款，无实质损失）。
*   **提示词工程**: 将召回的规则、法条、条款上下文动态注入 Prompt，约束 LLM 的生成范围。
*   **深度反思**: 对高/中风险触发二次审查，支持「维持/调级/存疑」三种结论。

### 7. 风险报告展示
*   **结构化输出**: 生成 Markdown 格式的详细报告，包含风险等级、风险点摘要、法条依据、修改建议。
*   **可视化**: 提供风险概览仪表盘，支持按高/中/低风险等级筛选查看。
*   **Word 导出**: 可一键导出 Word 格式报告。

---

## 技术栈

| 模块 | 技术选型 | 说明 |
| :--- | :--- | :--- |
| **前端框架** | **Reflex** | 全栈 Python Web 框架，编译为 React SPA，提供极致交互体验 |
| **后端服务** | **FastAPI** | 高性能异步 API 框架，承载核心业务逻辑 |
| **任务队列** | **BackgroundTasks** | FastAPI 原生后台任务，处理长耗时分析 |
| **大模型框架** | **LangChain** | 负责 Prompt 编排、模型调用接口封装 |
| **本地推理** | **Ollama** | 运行 qwen3:4b-instruct / DeepSeek 等开源大模型 |
| **LLM 调用** | **requests** | 直接调用 Ollama REST API，避免 httpx 兼容性问题 |
<<<<<<< HEAD
| **向量模型** | **BGE-small-zh-v1.5** | 中文语义向量表征 (Sentence-Transformers) |
| **重排序模型** | **BGE-Reranker-v2-m3** | 对检索结果进行二次精排，过滤低相关性候选 |
=======
| **向量模型** | **BGE-M3** | 中文语义向量表征 (Sentence-Transformers) |
>>>>>>> a24aa9bc36aad0e9578e26f7911cfdc68d1975ee
| **全文检索** | **Rank-BM25** | 传统的词袋模型检索，保证关键词命中率 |
| **数据存储** | **SQLite** | 轻量级关系型数据库，存储法条与历史记录 |

---

## 消融实验评测系统

系统内置了完整的消融实验框架，用于量化各组件对最终效果的贡献。主要用于项目继续改进的评估。

### 评测模式
以下四种模式均规定了输出格式便于解析。无Prompt指的是没有角色引导、先验知识等的注入。

| 模式 | 描述 | 组件 |
|------|------|------|
| **Mode 1** | 纯 LLM (Raw) | 无 Prompt 模板，直接输入 |
| **Mode 2** | 基础 Prompt | 格式化 Prompt，无规则引擎 |
| **Mode 3** | 当前工作流 | Prompt + 规则引擎 |
| **Mode 4** | 优化工作流 | CoT Prompt + 规则引擎 |

### 评测指标

| 指标 | 描述 |
|------|------|
| `accuracy` | 风险等级预测正确率 |
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) |
| `f1` | Precision 与 Recall 调和平均 |
| `hallucination_rate` | 证据不在原文中的比例 |
| `risk_id_accuracy` | 正例样本风险识别准确率 |
| `avg_latency_sec` | 平均响应时间（秒） |

### 运行评测

```bash
# 单模式评测
python evaluation/ablation_benchmark.py --mode 2 --limit 10

# 完整消融实验
python evaluation/ablation_benchmark.py --limit 20
```

> 详细技术文档请见：[TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md)

以上操作均可通过start_reflex.py脚本启动后在Web端进行。

## 快速开始-How to start

### 1. 环境准备
确保已安装 Python 3.10+ 和 Git。

```bash
# 克隆项目代码
git clone https://github.com/your-repo/contract-ai-agent.git
cd contract-ai-agent

# 创建并激活虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 模型准备
#### A. LLM 模型 (Ollama)
下载并安装 [Ollama](https://ollama.ai/)，然后拉取模型：
```bash
# 默认模型（推荐）
ollama pull qwen3:4b-instruct

# 或者使用大参数模型
ollama pull qwen2.5:14b
```

#### B. Embedding 模型
下载 `BAAI/bge-small-zh-v1.5` 模型到本地，并在 `configs/config.yaml` 中配置路径：
```yaml
embedding_config:
  model_path: "/path/to/your/bge-small-zh-v1.5"
```
*(注：也可以简单使用 HuggingFace 自动下载，无需手动配置路径)*

### 4. 启动系统
使用项目根目录下的启动脚本，一键拉起后端 API 和前端界面（启动完成后可自动跳转）：

```bash
python start_reflex.py
```

*   **访问前端**: 打开浏览器访问 `http://localhost:3000`
*   **查看文档**: 后端 API 文档位于 `http://localhost:8002/docs`

---

## 核心配置说明 (`configs/config.yaml`)

| 配置项 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `system_config.max_concurrency` | `5` | 并发分析的条款数量，显存越大可调越大 |
| `llm_config.model_name` | `qwen3:4b-instruct` | 使用的 LLM 模型名称，需与 Ollama 一致 |
| `llm_config.temperature` | `0` | 模型生成的随机度，法律场景建议低值以保准确 |
| `hybrid_search_config.threshold` | `0.75` | 规则匹配的最低置信度阈值 |
| `hybrid_search_config.alpha` | `0.65` | 混合检索权重系数 (0.65 代表 65% 语义 + 65% 关键词) |

---

## 项目结构

```
Contract-AI-Agent/
├── configs/                 # 配置文件 (模型参数、规则阈值)
<<<<<<< HEAD
│   ├── config.yaml          # 主配置
│   └── risk_standards.yaml  # 风险评估标准
├── data/                    # 数据资源
│   └── databases/           # 向量数据库与法条库
├── docs/                    # 文档目录
│   ├── weekly_reports/      # 周报
│   ├── reports/             # Word 报告
│   └── *.md                 # 技术文档
├── evaluation/              # 评测模块
│   ├── datasets/sources/    # 评测源数据 (GENERAL/LABOR/LEASE/SALES)
│   ├── results/             # 评测结果输出
│   ├── ablation_benchmark.py  # 消融实验主脚本
│   └── run_benchmark.py     # 评测运行器
=======
├── data/                    # 数据库与静态资源
├── evaluation/              # 评测脚本与数据集 (Benchmark)
>>>>>>> a24aa9bc36aad0e9578e26f7911cfdc68d1975ee
├── reflex_web/              # Reflex 前端
│   ├── reflex_web/
│   │   ├── components/      # UI 组件 (卡片、侧边栏、报告视图)
│   │   ├── state.py         # 前端状态管理 (AppState)
│   │   ├── styles.py        # 样式定义 (Glassmorphism)
│   │   └── reflex_web.py    # 页面入口 (Index, Report)
│   └── assets/              # 静态资源图片
├── scripts/                 # 工具脚本
│   ├── import_processed_laws.py  # 法条导入
│   ├── process_law_text.py       # 法条处理
│   └── test_llm.py               # LLM 测试
├── src/                     # 核心源码
│   ├── api/                 # FastAPI 后端接口
│   ├── core/                # 核心业务逻辑
│   │   ├── engine.py        # 分析主引擎
│   │   ├── rule_engine.py   # 规则匹配引擎
│   │   ├── llm.py           # LLM 调用封装
│   │   ├── hybrid_searcher.py # 混合检索器
│   │   ├── reference_retriever.py # 统一检索器 (Top-K + Rerank)
│   │   ├── reranker.py      # BGE-Reranker 重排序模块
│   │   ├── prompts.py       # Prompt 模板集中管理
│   │   ├── output_parser.py # LLM 输出解析器
│   │   ├── contract_classifier.py # 合同类型分类器
│   │   └── ollama_client.py # Ollama API 客户端
│   ├── database/            # 数据库操作
│   └── utils/               # 通用工具函数
├── .gitignore
├── README.md
├── requirements.txt
├── start_reflex.py          # 启动脚本
└── start_services.py        # 服务启动脚本
```

---

## 相关文档

| 文档 | 描述 |
|------|------|
| [TECHNICAL_DOCUMENTATION.md](./TECHNICAL_DOCUMENTATION.md) | 完整技术架构文档 |
| [evaluation/BENCHMARK_GENERATION_GUIDE.md](./evaluation/BENCHMARK_GENERATION_GUIDE.md) | 评测数据集生成指南 |

---

