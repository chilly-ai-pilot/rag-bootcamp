# Precision@K 功能说明

## 概述

本次更新为 RAG 评估系统添加了 **Precision@K** 指标，用于衡量检索结果的精确度。

## 什么是 Precision@K？

Precision@K 衡量前 K 个检索结果中有多少是相关的：

```
Precision@K = (前K个结果中相关文档的数量) / K
```

**示例：**
- 如果前5个检索结果中有3个包含答案片段，则 Precision@5 = 3/5 = 0.6
- 如果前5个检索结果中有1个包含答案片段，则 Precision@5 = 1/5 = 0.2

## 与其他指标的对比

| 指标 | 含义 | 适用场景 |
|------|------|----------|
| **Recall@K** | 是否在前K个中找到答案 | 评估检索覆盖率（0或1） |
| **Precision@K** | 前K个中有多少相关结果 | 评估检索精确度（0-1之间） |
| **MRR** | 答案排名的倒数平均值 | 评估答案排序质量 |

## 修改内容

### 1. scoring.py

新增函数 `calculate_precision_at_k()`：

```python
def calculate_precision_at_k(results: List[Dict], k: int) -> Dict[str, float]:
    """计算 Precision@K - 前K个检索结果中相关文档的比例"""
    # 按类别分桶收集精确度值
    # 统计前K个结果中有多少个与答案重叠
    # 返回每个类别的平均精确度
```

**关键逻辑：**
- 遍历每个查询的前 K 个检索结果
- 检查每个结果是否与 ground truth 答案区间重叠
- 计算相关结果的比例
- 按类别聚合并计算总体平均值

### 2. run_eval.py

#### 新增参数

```python
ap.add_argument("--precision-top-k", type=int, default=5, 
                help="计算 Precision@K 时考虑的前K个结果（默认5）")
```

#### 计算和显示

```python
# 计算 Precision@K
precision_scores = calculate_precision_at_k(results, args.precision_top_k)

# 打印结果
print(f"\n=== Precision@{args.precision_top_k} ===")
for cat, score in precision_scores.items():
    print(f"  {cat:24s} {score:.4f}")
```

#### 保存到 JSON

```python
result_data = {
    "metadata": {
        "model_config": {
            "precision_top_k": args.precision_top_k,
            # ...
        }
    },
    "precision_scores": precision_scores,
    # ...
}
```

#### 数据完整性

在 `batch_generate()` 函数中，确保每个结果包含 ground truth 信息：

```python
result = {
    "doc_id": q["doc_id"],
    "char_start": q["char_start"],
    "char_end": q["char_end"],
    # ... 其他字段
}
```

### 3. generate_report.py

#### 新增 Precision@K 卡片

在 HTML 报告中新增一个独立的 Precision@K 卡片，显示：
- 当前值
- 与上次评估的对比（diff）

#### 趋势图表

在"Metrics Trend Over Time"图表中添加 Precision@K 曲线：

```javascript
{
    label: 'Precision@K',
    data: [...],
    borderColor: 'rgb(255, 99, 132)',
    backgroundColor: 'rgba(255, 99, 132, 0.1)'
}
```

#### 历史表格

在评估历史表格中新增 Precision@K 列。

### 4. GitHub Actions (.github/workflows/eval_pipeline.yml)

#### 新增可配置参数

```yaml
workflow_dispatch:
  inputs:
    retrieval_top_k:
      description: 'Retrieval top-k (number of candidates for rerank)'
      default: '40'
      type: string
    
    rerank_top_k:
      description: 'Rerank top-k (number of results to send to generator)'
      default: '5'
      type: string
    
    precision_top_k:
      description: 'Precision@K top-k (number of results to calculate precision)'
      default: '5'
      type: string
```

#### 使用参数

```bash
RETRIEVAL_TOP_K="${{ github.event.inputs.retrieval_top_k || '40' }}"
RERANK_TOP_K="${{ github.event.inputs.rerank_top_k || '5' }}"
PRECISION_TOP_K="${{ github.event.inputs.precision_top_k || '5' }}"

python run_eval.py \
  --retrieval-top-k "$RETRIEVAL_TOP_K" \
  --rerank-top-k "$RERANK_TOP_K" \
  --precision-top-k "$PRECISION_TOP_K" \
  # ...
```

## 使用方法

### 本地运行

```bash
cd iteration7

# 使用默认 Precision@5
python run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode hybrid \
  --rerank-mode bge

# 自定义 Precision@K 值
python run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode hybrid \
  --rerank-mode bge \
  --precision-top-k 10
```

### GitHub Actions 手动触发

1. 进入 Actions 页面
2. 选择 "RAG Evaluation Pipeline"
3. 点击 "Run workflow"
4. 配置参数：
   - **Retrieval top-k**: 召回候选数量（默认 40）
   - **Rerank top-k**: 重排序后返回数量（默认 5）
   - **Precision top-k**: 计算精确度的前K个结果（默认 5）
5. 点击 "Run workflow" 启动评估

### 查看结果

#### 命令行输出

```
=== Precision@5 ===
  factual                  0.6200
  procedural               0.5800
  comparative              0.5400
  overall                  0.5800
```

#### HTML 报告

访问生成的 HTML 报告（`docs/index.html`）查看：
- Precision@K 卡片（与其他指标并列）
- 历史趋势图（Precision@K 曲线）
- 评估历史表格（包含 Precision@K 列）

#### JSON 结果文件

```json
{
  "metadata": {
    "model_config": {
      "precision_top_k": 5
    }
  },
  "precision_scores": {
    "factual": 0.62,
    "procedural": 0.58,
    "comparative": 0.54,
    "overall": 0.58
  }
}
```

## 测试

运行测试脚本验证功能：

```bash
cd iteration7
python3 test_precision.py
```

预期输出：

```
============================================================
Testing Precision@K Calculation
============================================================
✅ Precision@5 Results:
  test: 0.3000
  overall: 0.3000
  ✅ Test PASSED!
...
✅ All tests passed!
============================================================
```

## 注意事项

1. **K 值选择**：
   - `precision_top_k` 通常设置为 `rerank_top_k` 的值
   - 默认都是 5，确保评估的是实际送给生成器的结果

2. **数据依赖**：
   - Precision@K 需要 ground truth 信息（doc_id, char_start, char_end）
   - 确保查询集（queries.json）包含这些字段

3. **解读指标**：
   - Precision@K 高：检索结果精准，噪声少
   - Precision@K 低：检索结果中不相关文档多
   - 配合 Recall@K 和 MRR 综合评估检索质量

## 版本兼容性

- 兼容 Iteration 7 所有现有功能
- 向后兼容：旧的结果文件会显示 Precision@K = 0（因为缺少数据）
- 新生成的结果文件会包含完整的 Precision@K 数据

## 更新日期

2026-09-18
