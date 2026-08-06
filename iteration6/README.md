# Iteration 6: 引用验证机制 (Citation Validation)

## 目标

解决 Iteration 5 中发现的**引用幻觉问题** (Citation Hallucination)，通过结构化输出和三层验证机制，确保生成答案中的每个引用都有充分依据。

## 核心问题

Iteration 5 的评估分析发现了 3 类引用问题：

1. **标签幻觉**：LLM 编造不存在的片段编号（如 [文档13:片段999]）
2. **内容错配**：引用标签存在，但标注位置的内容并非来自该片段
3. **跨文档混淆**：将不同文档的信息混合后标注单一来源

这些问题导致 Faithfulness 分数下降（平均 0.938，但低分案例为 0.65-0.75）。

## 解决方案

### 1. 结构化输出 (Structured Output)

**旧版本** (Iteration 5):
```
答案: ST-500安装高度为1.8-2.2米[文档13:片段1]，步骤是...
```
- 问题：标注和内容混在一起，难以验证

**新版本** (Iteration 6):
```json
{
  "answer": "ST-500安装高度为1.8-2.2米，步骤是先按压粘贴再连接电源。",
  "citations": [
    {"span": "安装高度为1.8-2.2米", "source": "文档13:片段1"},
    {"span": "先按压粘贴再连接电源", "source": "文档13:片段2"}
  ]
}
```
- 优势：answer 和 citations 分离，span 必须是 answer 的子串

### 2. 三层验证机制

#### 层级 1: Span 存在性验证
```python
if span not in answer:
    reason = "span不在answer中"
```
- 确保 span 是 answer 的完整子串（逐字匹配）
- 追踪位置占用，避免重复标注

#### 层级 2: Source 合法性验证
```python
if source not in valid_sources:
    reason = "source不存在于合法来源列表"
```
- 在 prompt 中明确列出所有可用标签
- 拒绝 LLM 编造的标签

#### 层级 3: 内容一致性验证

使用梯度验证策略：

**3.1 精确匹配**（最可靠）
```python
if span in source_text:
    return True
```

**3.2 数字上下文验证**（防止张冠李戴）
```python
numbers_match_context(span, source_text, window=15)
```
- 错误示例：span "电压1.8V" 引用 source "安装高度1.8米"
- 正确示例：span "安装高度1.8米" 引用 source "安装高度1.8-2.2米"

**3.3 词汇重叠度验证**
```python
coverage = len(span_terms & source_terms) / len(span_terms)
return coverage >= 0.5
```
- 使用 2-gram 提取中文关键词（避免分词边界问题）
- 提取型号和数字（如 ST-500, 1.8-2.2）

### 3. 渲染策略

**从后往前插入标注**，避免位置偏移：
```python
for item in sorted(passed, key=lambda x: x["pos"], reverse=True):
    insert_pos = item["pos"] + len(item["span"])
    label = f"[{item['source']}]"
    final_answer = final_answer[:insert_pos] + label + final_answer[insert_pos:]
```

### 4. 失败处理

测试阶段：**记录所有验证失败案例**到 JSON，供人工分析

```json
{
  "query_id": "Q5",
  "validation": {
    "failed": [
      {
        "span": "工作电压为36V",
        "source": "文档13:片段1",
        "reason": "span内容在source原文中找不到充分依据（数字上下文不匹配）"
      }
    ],
    "stats": {
      "total": 3,
      "passed": 2,
      "failed": 1,
      "pass_rate": 0.667
    }
  }
}
```

后续可根据失败率决定处理策略：
- 失败率低 (<5%)：删除失败的 citations
- 失败率高 (>20%)：警告用户或重新生成

## 技术亮点

### Claude 测试代码启发

1. **2-gram 滑窗**：避免中文分词不一致
   ```python
   chinese_only = re.sub(r'[^\u4e00-\u9fa5]', '', text)
   bigrams = {chinese_only[i:i+2] for i in range(len(chinese_only) - 1)}
   ```

2. **数字上下文窗口**：防止数字张冠李戴
   ```python
   ctx = source_text[max(0, m.start() - window): m.end() + window]
   if extract_terms(ctx) & span_terms:
       ok = True
   ```

3. **位置占用追踪**：避免重复标注
   ```python
   occupied.append((pos, pos + len(span)))
   ```

### Prompt 改进

**旧版本**：没有列出可用标签，LLM 会瞎编
```
回答时用[文档X:片段N]标注引用来源...
```

**新版本**：明确列出所有合法标签
```
**可用的引用标签（请只使用这些标签，不要编造）**：
[文档13:片段1], [文档13:片段2], [文档5:片段3], [文档5:片段4]
```

## 文件结构

```
iteration6/
├── citation_validator.py  # 引用验证模块（新增）
├── generation.py          # 生成模块（新增 V6 函数）
├── run_eval.py           # 评估脚本（集成验证）
├── results_v6.json       # 评估结果（包含验证信息）
└── README.md             # 本文档
```

## 使用方法

### 1. 测试引用验证器

```bash
python3 citation_validator.py
```

预期输出：
```
✅ 通过验证: 2/3
❌ 未通过: 1/3
📊 通过率: 66.7%

最终答案（带标注）:
ST-500人体传感器安装高度为1.8-2.2米[文档13:片段1]，安装步骤是先按压粘贴，再连接电源完成配网[文档13:片段2]。工作电压为36V。

未通过验证的引用:
  ❌ '工作电压为36V' → 文档13:片段1
     原因: span内容在source原文中找不到充分依据（词汇重叠度不足或数字上下文不匹配）
```

### 2. 运行完整评估

```bash
# 使用新版本生成函数（带验证）
python3 run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode hybrid \
  --rerank-mode bge \
  --judge-mode llm \
  --generation-version v6
```

### 3. 分析验证失败率

```python
import json

with open('results_v6.json') as f:
    results = json.load(f)

failed_queries = [
    r for r in results['results']
    if r.get('validation', {}).get('stats', {}).get('failed', 0) > 0
]

print(f"有验证失败的查询: {len(failed_queries)}/{len(results['results'])}")
for r in failed_queries:
    print(f"\nQuery {r['id']}: {r['query']}")
    for fail in r['validation']['failed']:
        print(f"  ❌ '{fail['span']}' → {fail['source']}")
        print(f"     {fail['reason']}")
```

## 预期改进

基于 Iteration 5 的低分案例分析：

| Query | 问题类型 | Iteration 5 Faith | 预期 Iteration 6 Faith |
|-------|---------|-------------------|----------------------|
| Q5    | 标签错误 | 0.65              | 0.85+ （拒绝错误引用） |
| Q22   | 跨文档混淆 | 0.75              | 0.80+ （分别标注来源） |
| Q23   | 过度泛化 | 0.75              | 0.80+ （精确匹配原文） |

**总体目标**：
- Faithfulness 平均分从 0.938 提升至 0.95+
- 引用幻觉率从 ~10% 降低至 <5%

## 成本优化

- 使用 JSON 模式：无需额外 token 解析引用
- 批量验证：在内存中完成，无额外 API 成本
- 验证失败处理：只在测试阶段记录，生产环境可自动过滤

## 下一步

1. ✅ 实现 citation_validator.py
2. ✅ 更新 generation.py（V6 版本）
3. ⏳ 更新 run_eval.py（集成验证）
4. ⏳ 运行完整评估（32 queries）
5. ⏳ 分析验证失败率
6. ⏳ 对比 Iteration 5 vs Iteration 6 的 Faithfulness 分数
7. ⏳ 决定失败处理策略（删除 vs 警告 vs 重新生成）

## 参考资料

- Iteration 5 评估报告: `iteration5/judge_analysis_report.md`
- Claude 测试代码启发: 见本文档 "技术亮点" 部分
- 低分案例分析: Query 5 (0.65), Query 22 (0.75), Query 23 (0.75)
