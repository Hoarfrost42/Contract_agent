# Benchmark 测试用例生成指南

## 用于外部 LLM 的 Prompt 模板

```
你是一个合同法律专家，请根据以下风险规则，生成真实、自然的合同条款测试用例。

### 任务要求
1. 生成的条款应模拟真实合同场景，语言专业流畅
2. 每个条款 50-200 字
3. 正例必须包含指定风险，负例不应包含任何风险
4. 按照指定的 JSON 格式输出

### 目标风险规则
- risk_id: {风险ID}
- risk_name: {风险名称}
- keywords: {关键词列表}
- analysis_logic: {风险分析逻辑}
- laws: {相关法律}

### 输出格式
请严格按照以下 JSON 格式输出：
[
  {
    "id": "唯一ID",
    "contract_text": "合同条款文本",
    "expected_risks": [
      {
        "risk_id": "风险ID",
        "risk_name": "风险名称"
      }
    ],
    "case_type": "positive/negative/boundary"
  }
]

### 生成要求
- 正例 2 条：条款中明确包含该风险
- 负例 1 条：同类场景但不构成风险
- 边界用例 1 条：模糊情况，可能构成也可能不构成风险
```

## 批量生成的规则输入格式

将 `risk_rules.json` 中的规则按以下格式传给 LLM：

```json
{
  "rules_to_generate": [
    {
      "risk_id": "GENERAL_002",
      "risk_name": "单方解释权",
      "contract_type": "通用条款",
      "risk_type": "legal",
      "keywords": ["最终解释权", "解释权归公司", "以商家解释为准"],
      "analysis_logic": ""最终解释权"是典型的无效格式条款。合同解释应遵循法律规定，任何一方不得垄断解释权以规避责任。",
      "laws": "《民法典》第四百九十八条"
    }
  ],
  "output_per_rule": {
    "positive": 2,
    "negative": 1,
    "boundary": 1
  }
}
```

## 期望输出的最终 JSON 数组格式

```json
[
  {
    "id": "GENERAL_002_pos_001",
    "contract_text": "本合同的最终解释权归甲方所有，乙方对条款的理解应以甲方解释为准。如双方对合同内容产生歧义，以甲方的解释为最终裁定，乙方不得提出异议。",
    "expected_risks": [
      {
        "risk_id": "GENERAL_002",
        "risk_name": "单方解释权",
        "risk_type": "legal",
        "contract_type": "通用条款",
        "laws": "《民法典》第四百九十八条"
      }
    ],
    "case_type": "positive"
  },
  {
    "id": "GENERAL_002_pos_002",
    "contract_text": "有关本协议条款的含义，由公司享有独家解释权。用户在使用服务过程中产生的任何疑问，均以公司官方答复为准，用户应无条件接受。",
    "expected_risks": [
      {
        "risk_id": "GENERAL_002",
        "risk_name": "单方解释权",
        "risk_type": "legal",
        "contract_type": "通用条款",
        "laws": "《民法典》第四百九十八条"
      }
    ],
    "case_type": "positive"
  },
  {
    "id": "GENERAL_002_neg_001",
    "contract_text": "本合同条款如有争议，双方应友好协商解决；协商不成的，可向合同签订地人民法院提起诉讼。合同的解释应遵循公平原则和交易习惯。",
    "expected_risks": [],
    "case_type": "negative"
  },
  {
    "id": "GENERAL_002_boundary_001",
    "contract_text": "对于本合同中未尽事宜或条款理解存在分歧的，甲方有权作出解释说明，但该解释不影响乙方依法享有的权利。",
    "expected_risks": [],
    "case_type": "boundary",
    "notes": "边界：有解释权但保留乙方法定权利，需人工判断"
  }
]
```

## 质量检查清单

生成后请确保：
- [ ] 所有 `id` 唯一且格式统一
- [ ] 正例的 `expected_risks` 非空
- [ ] 负例的 `expected_risks` 为 `[]`
- [ ] `case_type` 仅为 `positive`/`negative`/`boundary`
- [ ] 条款内容自然真实，不是简单拼凑关键词
