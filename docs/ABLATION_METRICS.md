# 消融实验评测指标文档

本文档详细说明消融实验 (`evaluation/ablation_benchmark.py`) 中使用的所有评测指标及其计算公式。

---

## 一、风险等级评估指标

### 1.1 Accuracy (精确准确率)

```
Accuracy = correct_risk / total
```

- 仅精确匹配计分：高=高、中=中、低=低
- 差一级或两级均视为错误

### 1.2 Weighted Accuracy (非对称加权准确率) ⭐ 已更新

```
Weighted_Accuracy = Σ weighted_score / total
```

**非对称加权评分规则**（2024.01 更新）：

| Ground Truth | Prediction | Score | 理由 |
|--------------|------------|-------|------|
| 高 | 高 | 1.0 | 精确匹配 |
| 高 | 中 | **0.4** | 风险降级，危险行为 |
| 高 | 低 | 0.0 | 致命漏判，零容忍 |
| 中 | 高 | **0.8** | 过度谨慎，可接受 |
| 中 | 中 | 1.0 | 精确匹配 |
| 中 | 低 | 0.4 | 有一定风险 |
| 低 | 高 | 0.5 | 过度报警 |
| 低 | 中 | **0.8** | 防御性误判 |
| 低 | 低 | 1.0 | 精确匹配 |

> **设计理念**：在风控场景下，"漏报"远比"虚警"严重。因此对"风险降级"行为（High→Medium, High→Low）给予更重的扣分。

### 1.3 Linear Weighted Kappa (LWK) ⭐ 推荐使用

```
Kappa = 1 - Σ(W × O) / Σ(W × E)

线性权重公式: w_ij = |i - j| / (N - 1)
```

**线性权重矩阵 W (LWK)**：

```
           pred高  pred中  pred低
gt高      [ 0.0,   0.5,   1.0  ]
gt中      [ 0.5,   0.0,   0.5  ]
gt低      [ 1.0,   0.5,   0.0  ]
```

> **为什么用 LWK 替代 QWK**：QWK 使用二次方惩罚，对"离群"错误惩罚极重（错两档惩罚是错一档的4倍），导致 Kappa 值极低。LWK 使用线性惩罚，对有序分类更稳健。

### 1.4 High-Risk F2-Score ⭐ 新增

```
F2 = (1 + β²) × (P × R) / (β² × P + R)，其中 β = 2
```

- F2 分数中 Recall 权重是 Precision 的 2 倍
- 适合"宁可错杀，不可漏过"的风控场景
- 仅针对 High 风险类别计算

### 1.5 Macro F1 / Precision / Recall

对高、中、低三类分别计算 F1，然后取平均：

```
Macro_F1 = (F1_高 + F1_中 + F1_低) / 3

# 单类计算
TP = conf_matrix[k][k]
FP = Σ conf_matrix[i][k] - TP  (其他类被误判为 k)
FN = Σ conf_matrix[k][j] - TP  (k 类被误判为其他)

Precision_k = TP / (TP + FP)
Recall_k = TP / (TP + FN)
F1_k = 2 × P × R / (P + R)
```

### 1.5 Class F1 (分类别 F1)

除了 Macro F1 外，还输出每个风险等级的单独 F1 分数：

```json
"class_f1": {
    "High": 0.75,    // 高风险类别的 F1
    "Medium": 0.62,  // 中风险类别的 F1
    "Low": 0.88      // 低风险类别的 F1
}
```

用于分析模型在哪个风险等级上表现较弱。

---

## 二、幻觉检测指标

### 2.1 Hallucination Rate (综合幻觉率)

```
Hallucination_Rate = evidence_invalid / (evidence_valid + evidence_invalid)
```

### 2.2 Clause Hallucination Rate (条款证据幻觉率)

```
Clause_Hallucination = clause_invalid / (clause_valid + clause_invalid)
```

检测 LLM 输出的"证据摘录"是否真实存在于原条款中。

### 2.3 Law Hallucination Rate (法条引用幻觉率)

```
Law_Hallucination = law_invalid / (law_valid + law_invalid)
```

检测 LLM 引用的法律条文是否真实存在。

**两阶段检测流程** (`verify_evidence`):

1. **Stage 1**: Embedding 召回 (BGE-Small) → 选出 Top-3 候选句子
2. **Stage 2**: Reranker 精排 (BGE-Reranker) → 分数 ≥ threshold 则有效

---

## 三、规则触发一致性指标

### 3.1 Rule Recall (规则召回率)

```
Rule_Recall = rule_correct / rule_target
```

- `rule_target`: Ground Truth 中标注的应触发规则数
- `rule_correct`: 系统正确触发的规则数

### 3.2 Rule Precision (规则精确率)

```
Rule_Precision = rule_correct / rule_trigger
```

- `rule_trigger`: 系统实际触发的规则数

### 3.3 Risk ID Accuracy

```
Risk_ID_Accuracy = risk_id_match / risk_id_total
```

检测检索到的 `risk_id` 是否与 Ground Truth 标注一致。

---

## 四、综合任务指标

### 4.1 Task Success Rate (任务成功率)

```
Task_Success_Rate = success_count / total
```

完全成功需同时满足：
- 解析成功 (`parse_success`)
- 风险等级正确
- 证据有效 (非幻觉)
- 有修改建议

### 4.2 Parse Rate (解析成功率)

```
Parse_Rate = parse_success / total
```

LLM 输出是否符合预期的 Markdown 格式。

### 4.3 Avg Latency (平均响应时间)

```
Avg_Latency = total_latency / total  (秒)
```

---

## 五、混淆矩阵

三分类混淆矩阵结构：

```
                预测
               高     中     低
       高   [ TP_H, E_HM, E_HL ]
实际   中   [ E_MH, TP_M, E_ML ]
       低   [ E_LH, E_LM, TP_L ]
```

- **对角线**: 正确分类数
- **非对角线**: 错误分类数

用于生成 Heatmap 可视化图表。

---

## 代码位置

所有指标定义于 `evaluation/ablation_benchmark.py` 的 `EvalMetrics` 类 (L130-L360)。
