# 消融实验评测指标文档

本文档详细说明消融实验图表生成器 (`chart_generator.py`) 中使用的核心评测指标及其计算公式。

---

## 一、核心性能指标 (The Safety & Logic Bar)

### 1.1 High-Risk F2-Score ⭐ 主指标

```
F2 = (1 + β²) × (P × R) / (β² × P + R)，其中 β = 2
```

- **侧重 Recall**：F2 分数中 Recall 权重是 Precision 的 2 倍。
- **设计理念**：在风控场景下，"宁可错杀，不可漏过"。
- **计算范围**：仅针对 **High** 风险类别计算。

### 1.2 Quadratic Weighted Kappa (QWK) ⭐ 逻辑指标

```
Kappa = 1 - Σ(W × O) / Σ(W × E)

二次方权重公式: w_ij = (i - j)² / (N - 1)²
```

**二次方权重矩阵 W (QWK)**：

```
           pred高  pred中  pred低
gt高      [ 0.0,   0.25,  1.0  ]
gt中      [ 0.25,  0.0,   0.25 ]
gt低      [ 1.0,   0.25,  0.0  ]
```

- **惩罚差异**：对"离群"错误（如 High→Low）给予极重的惩罚（是相邻错误的4倍）。
- **作用**：衡量模型风险评级逻辑的一致性，惩罚严重的逻辑跳跃。

### 1.3 Weighted Accuracy (非对称加权准确率) ⭐ 落地指标

```
Weighted_Accuracy = Σ weighted_score / total
```

**非对称加权评分规则**：

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

### 1.4 Risk ID Precision (规则匹配精确率) ⭐ 可信度指标

```
Risk_ID_Precision = risk_id_correct / risk_id_total
```

- 检测检索到的 `risk_id` 是否与 Ground Truth 标注一致。
- 反映模型是否"找对了规则"，而不仅仅是蒙对了风险等级。

---

## 二、系统稳定性指标 (The Quality Check)

### 2.1 Hallucination Rate (综合幻觉率)

```
Hallucination_Rate = evidence_invalid / (evidence_valid + evidence_invalid)
```

- 检测 LLM 输出的"证据摘录"及"法条引用"是否真实存在。
- **目标**：越低越好。

### 2.2 Task Success Rate (任务成功率)

```
Task_Success_Rate = success_count / total
```

完全成功需同时满足：
1. 解析成功
2. 风险等级正确
3. 证据有效 (非幻觉)
4. 输出合规

### 2.3 High-Risk Leakage (高风险漏判率) ⭐ 风控红线

```
High_Risk_Leakage = Count(High → Medium) / Total(High)
```

- **定义**：真实为**高风险**的条款，被误判为**中风险**的比例。
- **High → Low** 通常被视为不可接受的错误（权重0），而 **High → Medium** 往往是模型为了"讨好"用户而做出的妥协，是风控系统中最隐蔽的漏洞。
- **目标**：越低越好。

---

## 三、行为映射矩阵 (Map)

### 3.1 混淆矩阵 (Confusion Matrix)

```
                预测
               高     中     低
       高   [ TP_H, E_HM, E_HL ]
实际   中   [ E_MH, TP_M, E_ML ]
       低   [ E_LH, E_LM, TP_L ]
```

- **可视化**：用于生成 Heatmap，直观展示模型的行为倾向（如是否过度防御、是否存在严重的漏判）。
