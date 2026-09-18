# Precision@K 快速开始指南

## 🚀 立即使用

### 1️⃣ 本地运行（使用默认参数）

```bash
cd iteration7
python3 run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode hybrid \
  --rerank-mode bge
```

**默认配置：**
- `--retrieval-top-k 40` （召回40个候选）
- `--rerank-top-k 5` （重排序后保留5个）
- `--precision-top-k 5` （计算前5个的精确度）

### 2️⃣ 自定义 Precision@K

```bash
python3 run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode hybrid \
  --rerank-mode bge \
  --precision-top-k 10
```

### 3️⃣ 完整参数控制

```bash
python3 run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode hybrid \
  --retrieval-top-k 100 \
  --rerank-mode bge \
  --rerank-top-k 10 \
  --precision-top-k 10 \
  --judge-mode deepseek
```

## 📊 查看结果

### 命令行输出

运行后会显示：

```
=== Precision@5 ===
  factual                  0.6200
  procedural               0.5800
  comparative              0.5400
  overall                  0.5800
```

### HTML 报告

```bash
# 生成 HTML 报告
python3 generate_report.py --data-dir ../data --output-dir ../docs

# 打开报告
open ../docs/index.html
```

在报告中你会看到：
- 🎴 Precision@K 卡片（与其他指标并列显示）
- 📈 历史趋势图（包含 Precision@K 曲线）
- 📋 评估历史表格（包含 Precision@K 列）

### JSON 结果文件

```bash
# 查看最新结果
cd ../data
ls -lt results_*.json | head -1
cat $(ls -t results_*.json | head -1) | jq '.precision_scores'
```

输出示例：
```json
{
  "factual": 0.62,
  "procedural": 0.58,
  "comparative": 0.54,
  "overall": 0.58
}
```

## 🤖 GitHub Actions 使用

### 手动触发工作流

1. 访问 GitHub 仓库的 **Actions** 页面
2. 选择 **"RAG Evaluation Pipeline"** 工作流
3. 点击 **"Run workflow"** 按钮
4. 配置参数：

   ```yaml
   Chunking strategy: fixed_100_50
   Retrieval mode: hybrid
   Retrieval top-k: 40          # ← 新参数
   Rerank mode: bge
   Rerank top-k: 5              # ← 新参数
   Precision top-k: 5           # ← 新参数（重点）
   Judge mode: none
   ```

5. 点击绿色的 **"Run workflow"** 按钮启动

### 查看结果

- 工作流完成后，会自动提交结果到 `data/` 目录
- HTML 报告会更新到 `docs/index.html`
- 可以通过 GitHub Pages 访问在线报告

## 🧪 测试功能

验证 Precision@K 计算是否正确：

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

## 📖 理解 Precision@K

### 什么是 Precision@K？

Precision@K 衡量前 K 个检索结果中有多少是相关的：

```
Precision@K = (前K个结果中相关文档数) / K
```

### 实际例子

假设查询"产品价格是多少？"，检索返回5个结果：

| 排名 | 文档 | 是否相关 |
|-----|------|---------|
| 1 | 价格说明段落 | ✅ 相关 |
| 2 | 产品功能介绍 | ❌ 不相关 |
| 3 | 价格表格 | ✅ 相关 |
| 4 | 联系方式 | ❌ 不相关 |
| 5 | 退款政策 | ❌ 不相关 |

**计算：**
- 前5个中有2个相关
- Precision@5 = 2/5 = **0.4**

### 为什么重要？

- ✅ **高 Precision**：检索结果精准，生成器看到的都是有用信息
- ❌ **低 Precision**：检索结果噪声多，影响答案质量

## 🎯 最佳实践

### 参数设置建议

1. **标准场景**（推荐）
   ```bash
   --retrieval-top-k 40
   --rerank-top-k 5
   --precision-top-k 5
   ```
   评估实际送给生成器的5个结果的质量

2. **高质量要求**
   ```bash
   --retrieval-top-k 100
   --rerank-top-k 10
   --precision-top-k 10
   ```
   增加候选池，评估更多结果

3. **快速测试**
   ```bash
   --retrieval-top-k 20
   --rerank-top-k 3
   --precision-top-k 3
   ```
   减少计算量，快速迭代

### 指标对比

| 指标 | 回答的问题 | 典型值 |
|------|-----------|--------|
| **Recall@40** | 40个结果中是否有答案？ | 0.85-0.95 |
| **Precision@5** | 前5个中有多少相关？ | 0.40-0.70 |
| **MRR** | 答案平均排在第几？ | 0.60-0.80 |

### 优化建议

- **Precision@5 < 0.5**：检索质量差，考虑：
  - 优化 embedding 模型
  - 调整 chunking 策略
  - 改进查询重写
  
- **Precision@5 > 0.7**：检索质量好，可以：
  - 减少 rerank_top_k（节省成本）
  - 专注优化生成质量

## 🆘 常见问题

### Q1: Precision@K 显示为 0？

**原因：** 旧的结果文件不包含 precision_scores 数据

**解决：** 重新运行评估生成新结果

### Q2: 如何只计算 Precision，不运行 Judge？

```bash
python3 run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode hybrid \
  --rerank-mode bge \
  --judge-mode none  # ← 关闭 Judge
```

### Q3: precision-top-k 应该设置为多少？

**建议：** 设置为 `rerank-top-k` 的值

**原因：** 评估实际送给生成器的结果质量

### Q4: 能否评估不同 K 值的 Precision？

**可以！** 多次运行评估，使用不同的 `--precision-top-k` 值：

```bash
# Precision@3
python3 run_eval.py ... --precision-top-k 3

# Precision@5
python3 run_eval.py ... --precision-top-k 5

# Precision@10
python3 run_eval.py ... --precision-top-k 10
```

## 📚 延伸阅读

- `PRECISION_FEATURE.md` - 详细功能说明
- `CHANGES_SUMMARY.md` - 完整变更列表
- `test_precision.py` - 测试代码和示例

## ✨ 快速总结

```bash
# 1. 运行评估（自动计算 Precision@5）
cd iteration7
python3 run_eval.py --chunking-strategy fixed_100_50 --retrieval-mode hybrid --rerank-mode bge

# 2. 生成报告
python3 generate_report.py --data-dir ../data --output-dir ../docs

# 3. 查看结果
open ../docs/index.html
```

就这么简单！🎉
