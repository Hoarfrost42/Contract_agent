# 合同风险监测智能体 - 优化指导文档

> **生成日期**：2025-12-25  
> **基于**：项目当前状态 + 外部优化建议  
> **目标**：指导后续可落地的改进工作

---

## 📋 优化建议与项目现状匹配分析

| 建议类别 | 项目现状 | 可行性评估 | 优先级 |
|---------|---------|-----------|:------:|
| **Prompt 输出改进** | 已有结构化 Prompt + 多模式正则解析 | ✅ 高度可行 | P0 |
| **检索体系提升** | 已有 BM25+Dense 混合检索 | ✅ 部分可行 | P1 |
| **LangGraph 迁移** | 已预留 state.py，工作流用 asyncio | ⚠️ 有限可行 | P2 |
| **多代理流程** | 本地性能受限 | ❌ 暂不推荐 | - |
| **评测指标升级** | 已有基础评测系统 | ✅ 可行 | P1 |
| **自我反思完善** | 已有 Self-Reflection 模式 | ✅ 可行 | P1 |

---

## 🎯 P0 优先级：Prompt 输出改进

### 现状分析

当前 `llm.py` 中的 Prompt 设计：
- ✅ 已有角色定位（"你是合同风险分析专家"）
- ✅ 已有输出格式约束（Markdown 模板）
- ⚠️ 未使用 Chain-of-Thought 分步推理
- ⚠️ 未使用 Few-Shot 示例教学

### 改进方案 1：引入 Chain-of-Thought 分步推理

**修改文件**：`src/core/llm.py` 中的 `MERGED_SCAN_PROMPT`

```python
# 改进前（当前）
"""分析以下合同条款的风险..."""

# 改进后（建议）
"""
请按以下步骤分析合同条款风险：

**第一步：关键要素识别**
- 识别条款中的主体（甲方/乙方）
- 识别核心动作（权利/义务/限制）
- 识别关键数字（金额/期限/比例）

**第二步：风险点定位**
- 对照参考信息中的专家知识
- 检查是否存在权利不对等
- 检查是否违反法律强制性规定

**第三步：综合评估输出**
- 给出风险等级（高/低）
- 提供证据摘录
- 给出修改建议

条款原文：
{clause}

参考信息：
{reference}
"""
```

**预期收益**：
- 提升分析透明度，便于用户理解 AI 推理过程
- 减少遗漏关键风险点的概率
- 为论文"可解释性"章节提供素材

### 改进方案 2：添加 Few-Shot 示例

**修改文件**：`src/core/llm.py`

```python
FEW_SHOT_EXAMPLES = """
【示例1】
条款：甲方有权在任何情况下单方解除合同，无需承担任何责任。
分析：
## 风险：单方解除权不对等
- **等级**：高
- **证据**：「甲方有权在任何情况下单方解除合同」
- **问题**：赋予甲方无条件解除权，乙方权益无保障
- **法条**：《民法典》第563条
- **建议**：增加解除条件限制，如"经双方协商一致"

【示例2】
条款：本合同自双方签字盖章之日起生效。
分析：
## 风险：无
- **等级**：低
- **原因**：标准生效条款，符合一般合同惯例
"""
```

**工作量**：0.5-1 天

---

## 🎯 P1 优先级：检索体系提升

### 现状分析

- ✅ 已实现 BM25 + Dense 混合检索
- ✅ 已有 SQLite 法条精确查询
- ⚠️ 规则库仅支持文本相似度匹配
- ⚠️ 无案例库语义检索

### 改进方案：增强 RAG 管道

**阶段1：优化现有检索（1-2天）**

| 改进项 | 说明 |
|-------|------|
| 检索结果重排序 | 添加 Reranker 对 Top-K 结果二次排序 |
| 阈值自适应 | 根据条款长度动态调整相似度阈值 |
| 检索日志 | 记录每次检索的召回规则，便于分析 |

**阶段2：引入向量数据库（1-2周，可选）**

```
当前: HybridSearcher (内存计算)
       ↓
改进: FAISS 本地向量索引
       ↓
       - 支持更大规则库
       - 支持案例库扩展
       - 支持增量更新
```

**建议**：阶段1短期可行，阶段2可作为论文"未来工作"

---

## 🎯 P1 优先级：评测指标升级

### 现状分析

当前 `ablation_benchmark.py` 指标：
- 风险等级准确率 (`accuracy`)
- 精确率/召回率/F1 (`precision`, `recall`, `f1`)
- 解析成功率 (`parse_rate`)
- 幻觉率 (`hallucination_rate`)
- 风险ID匹配率 (`risk_id_accuracy`)
- 平均响应时间 (`avg_latency_sec`)

### 改进方案

**已实现的评测指标**：

| 指标 | 计算方式 | 意义 |
|------|---------|------|
| **accuracy** | 正确预测数 / 总数 | 整体准确率 |
| **F1 分数** | `2 * P * R / (P + R)` | 综合精确率和召回率 |
| **幻觉率** | 证据不在原文中的比例 | 衡量虚构内容比例 |
| **parse_rate** | 解析成功数 / 总数 | LLM 输出格式稳定性 |

**代码修改示例**（`evaluation/run_benchmark.py`）：

```python
def calculate_f1(predictions, ground_truth):
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p == g == "高")
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p == "高" and g != "高")
    fn = sum(1 for p, g in zip(predictions, ground_truth) if p != "高" and g == "高")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}
```

**工作量**：1-2 天

---

## 🎯 P1 优先级：自我反思完善

### 现状分析

当前 `llm.py` 中的 `self_reflect()` 方法：
- ✅ 对高风险判定进行二次审查
- ⚠️ 仅审查，未形成闭环修正

### 改进方案

**阶段1：增强反思 Prompt**

```python
SELF_REFLECTION_PROMPT_V2 = """
请审查以下风险分析，检查是否存在问题：

**原始分析**：
{original_analysis}

**审查清单**：
1. [ ] 证据是否确实存在于原文中？
2. [ ] 风险等级是否与证据严重程度匹配？
3. [ ] 法条引用是否准确？
4. [ ] 是否存在遗漏的关键风险点？
5. [ ] 建议是否具有可操作性？

如发现问题，请输出修正后的分析。
如无问题，请输出"确认无误"。
"""
```

**阶段2：闭环修正**

```python
def analyze_with_reflection(self, clause, reference):
    # 第一轮：初步分析
    result = self.analyze_clause(clause, reference)
    
    if result.risk_level == "高":
        # 第二轮：反思审查
        reflection = self.self_reflect(result)
        
        if "修正" in reflection:
            # 第三轮：采纳修正
            result = self._apply_correction(result, reflection)
    
    return result
```

**工作量**：1-2 天

---

## 🎯 P2 优先级：LangGraph 有限迁移

### 现状分析

- 已有 `state.py` 定义状态类型
- 工作流使用 `asyncio.gather` 并行处理
- 本地 LLM 性能有限

### 可行方案：小范围试点

仅将**深度反思流程**改为 LangGraph 条件分支：

```python
from langgraph.graph import StateGraph, END

def create_reflection_graph():
    graph = StateGraph(AnalysisState)
    
    graph.add_node("analyze", analyze_clause_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("finalize", finalize_node)
    
    graph.add_conditional_edges(
        "analyze",
        lambda state: "reflect" if state["risk_level"] == "高" else "finalize"
    )
    graph.add_edge("reflect", "finalize")
    graph.add_edge("finalize", END)
    
    return graph.compile()
```

**收益**：
- 代码更清晰，逻辑可视化
- 为论文提供 LangGraph 使用案例
- 不影响整体性能

**工作量**：3-5 天

---

## ❌ 暂不推荐：多代理流程

### 原因

1. **本地性能限制**：多代理需要多次 LLM 调用，本地 4B 模型延迟较高
2. **复杂度增加**：调试和维护成本显著上升
3. **收益有限**：当前单 Agent + 规则引擎架构已能满足需求

### 替代方案

在论文中论述"多代理架构设计思路"作为未来工作，无需实际实现。

---

## 📅 实施计划建议

| 周次 | 任务 | 产出 |
|:----:|------|------|
| **第1周** | Prompt CoT + Few-Shot 改进 | 更新后的 `llm.py` |
| **第1周** | 评测指标升级 | F1/Jaccard 指标报告 |
| **第2周** | 自我反思闭环修正 | 增强版 `self_reflect()` |
| **第2周** | 检索日志 + 阈值优化 | 检索分析报告 |
| **第3周**（可选） | LangGraph 小范围试点 | 条件分支工作流 |

---

## 📝 论文素材建议

以上改进可为论文提供以下章节素材：

| 章节 | 素材 |
|------|------|
| **方法论** | CoT 分步推理、Few-Shot 示例教学 |
| **实验评估** | F1/Jaccard/幻觉率等指标对比 |
| **系统设计** | 混合检索 + 规则引擎 + LLM 验证架构 |
| **可解释性** | 分步推理过程展示 |
| **未来工作** | LangGraph 状态机、多代理协作、知识图谱 |
