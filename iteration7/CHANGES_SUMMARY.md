# Precision@K 功能变更总结

## 变更文件列表

### ✅ 已修改的文件

1. **`scoring.py`**
   - ✅ 新增 `calculate_precision_at_k()` 函数
   - ✅ 按类别计算前K个检索结果的精确度
   - ✅ 支持多类别聚合和总体统计

2. **`run_eval.py`**
   - ✅ 导入 `calculate_precision_at_k` 函数
   - ✅ 新增 `--precision-top-k` 参数（默认5）
   - ✅ 在 `batch_generate()` 中保存 ground truth 信息（doc_id, char_start, char_end）
   - ✅ 计算并打印 Precision@K 分数
   - ✅ 保存 Precision@K 到 JSON 结果文件
   - ✅ 更新函数返回值包含 precision_scores

3. **`generate_report.py`**
   - ✅ 在 `extract_metrics()` 中提取 precision_scores
   - ✅ 添加 Precision@K 卡片到 HTML 报告
   - ✅ 在趋势图中添加 Precision@K 曲线
   - ✅ 在历史表格中添加 Precision@K 列
   - ✅ 计算 Precision@K 的对比差异

4. **`.github/workflows/eval_pipeline.yml`**
   - ✅ 新增 `retrieval_top_k` 输入参数（默认40）
   - ✅ 新增 `rerank_top_k` 输入参数（默认5）
   - ✅ 新增 `precision_top_k` 输入参数（默认5）
   - ✅ 在运行评估时传递这些参数
   - ✅ 在配置日志中显示参数值

### ✅ 新增的文件

1. **`test_precision.py`**
   - 测试 Precision@K 计算功能
   - 验证计算逻辑正确性
   - 包含 Precision@5 和 Precision@3 测试案例

2. **`PRECISION_FEATURE.md`**
   - Precision@K 功能详细说明文档
   - 使用方法和示例
   - 与其他指标的对比

3. **`CHANGES_SUMMARY.md`**
   - 本文件，变更总结

## 功能验证

### ✅ Python 语法检查
```bash
cd iteration7
python3 -m py_compile run_eval.py      # ✅ 通过
python3 -m py_compile scoring.py        # ✅ 通过
python3 -m py_compile generate_report.py # ✅ 通过
```

### ✅ YAML 语法检查
```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/eval_pipeline.yml'))"
# ✅ 通过
```

### ✅ 单元测试
```bash
cd iteration7
python3 test_precision.py
# ✅ 所有测试通过
```

## 参数配置对照表

| 参数名称 | 命令行参数 | GitHub Actions 输入 | 默认值 | 说明 |
|---------|-----------|-------------------|--------|------|
| Retrieval Top-K | `--retrieval-top-k` | `retrieval_top_k` | 40 | 召回的候选数量（送入 Rerank） |
| Rerank Top-K | `--rerank-top-k` | `rerank_top_k` | 5 | Rerank 后返回的数量（送给 Generator） |
| **Precision Top-K** | `--precision-top-k` | `precision_top_k` | 5 | 计算 Precision@K 时考虑的前K个结果 |

## 数据流程

```
查询 Query
    ↓
检索 Retrieval (top-40)  ← 使用 --retrieval-top-k
    ↓
重排序 Rerank (top-5)    ← 使用 --rerank-top-k
    ↓
生成答案 Generation
    ↓
评估指标:
  - Recall@K (检索是否命中)
  - Precision@K (前K个中有多少相关) ← 使用 --precision-top-k
  - MRR (答案排名质量)
  - Faithfulness (答案忠实度)
  - Relevance (答案相关性)
```

## 输出示例

### 命令行输出
```
=== Recall@40 (chunking: fixed_100_50, retrieval: hybrid) ===
  factual                  0.92
  procedural               0.88
  comparative              0.85
  overall                  0.88

=== Precision@5 ===
  factual                  0.6200
  procedural               0.5800
  comparative              0.5400
  overall                  0.5800

=== MRR (Mean Reciprocal Rank) ===
  factual                  0.7854
  procedural               0.7234
  comparative              0.6987
  overall                  0.7358
```

### HTML 报告
- 📊 新增 Precision@K 卡片（显示当前值和对比差异）
- 📈 趋势图包含 Precision@K 曲线（粉色线条）
- 📋 历史表格新增 Precision@K 列

### JSON 结果文件
```json
{
  "metadata": {
    "model_config": {
      "retrieval_top_k": 40,
      "rerank_top_k": 5,
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

## 使用建议

### 典型配置组合

1. **标准评估**（推荐）
   ```bash
   --retrieval-top-k 40 --rerank-top-k 5 --precision-top-k 5
   ```
   - 召回40个候选
   - 重排序后取前5个
   - 计算前5个的精确度

2. **高召回评估**
   ```bash
   --retrieval-top-k 100 --rerank-top-k 10 --precision-top-k 10
   ```
   - 增加召回范围
   - 评估更多结果的质量

3. **快速评估**
   ```bash
   --retrieval-top-k 20 --rerank-top-k 3 --precision-top-k 3
   ```
   - 减少计算量
   - 快速迭代测试

### 参数设置建议

- `precision_top_k` 建议等于 `rerank_top_k`
  - 原因：评估实际送给生成器的结果质量
  
- `retrieval_top_k` >= `rerank_top_k` >= `precision_top_k`
  - 原因：保证逻辑一致性

## 向后兼容性

✅ **完全兼容**：
- 旧的评估结果文件仍可正常显示（Precision@K 显示为 0）
- 新参数都有默认值，不影响现有脚本
- HTML 报告会优雅处理缺失的 precision_scores 数据

## 后续工作（可选）

- [ ] 添加 Precision@K 到 iteration8
- [ ] 添加更多粒度的 Precision 分析（按 rerank 分数区间）
- [ ] 添加 Precision-Recall 曲线绘制
- [ ] 支持自定义相关性判断（不仅是重叠，还可以用相似度）

## 完成状态

🎉 **所有功能已完成并测试通过！**

- ✅ Precision@K 计算逻辑
- ✅ 命令行参数
- ✅ JSON 结果保存
- ✅ HTML 报告展示
- ✅ GitHub Actions 集成
- ✅ 单元测试
- ✅ 文档说明

---

**变更日期**: 2026-09-18  
**版本**: Iteration 7 - Precision@K Feature
