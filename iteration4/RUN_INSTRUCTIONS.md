# Iteration 4 运行说明

## 📦 前置准备

### 1. 安装依赖

```bash
cd iteration4
pip install -r requirements.txt
```

**关键依赖：**
- `FlagEmbedding>=1.2.0` - bge-reranker-base 模型

---

## 🚀 快速运行（推荐）

使用自动化脚本，一键完成所有步骤：

```bash
./run_iteration4.sh
```

脚本会自动：
1. ✅ 检查并安装依赖
2. ✅ 运行 reranker 功能测试
3. ✅ 运行 hybrid baseline（如果还没有）
4. ✅ 运行 rerank 评估
5. ✅ 生成结果文件

---

## 📝 手动运行（分步执行）

如果你想逐步执行，按以下顺序：

### Step 1: 测试 Reranker

```bash
python3 test_reranker.py
```

**预期输出：**
```
[Test 1] 检查依赖导入...
✅ FlagEmbedding 导入成功

[Test 2] 加载 bge-reranker-base 模型...
✅ Reranker 模型加载成功

[Test 3] 测试 Rerank 基本功能...
✅ 分数范围正常 (0-1)
✅ 排序逻辑正确

[Test 4] 测试与检索系统集成...
✅ 加载了 XXX 个文档块
✅ Hybrid 检索返回 5 个结果
✅ Rerank 检索返回 5 个结果

🎉 所有测试通过！
```

### Step 2: 运行 Hybrid Baseline（如果还没有）

```bash
python3 run_eval.py --chunking-strategy small_100_50 --retrieval-mode hybrid
```

**预期输出：**
```
=== Recall@5 ===
  overall                  0.78

=== MRR ===
  overall                  0.5703

✅ Results written to results_small_100_50_hybrid.json
```

### Step 3: 运行 Rerank 评估

```bash
python3 run_eval.py --chunking-strategy small_100_50 --retrieval-mode rerank
```

**预期输出：**
```
Loading bge-reranker-base model...
(第一次运行会下载模型，约 1.2GB，后续使用缓存)

=== Recall@5 ===
  overall                  0.90+

=== MRR ===
  overall                  0.80+

=== Rerank Score Distribution ===
  Total scores:            160
  Range:                   [0.XXXX, 0.XXXX]
  Mean:                    0.XXXX
  
  Suggested thresholds for Iteration 6:
    Conservative:          0.XXXX
    Recommended:           0.XXXX
    Aggressive:            0.XXXX

✅ Results written to results_small_100_50_rerank.json
```

---

## 📊 查看结果

### 生成的文件

```
iteration4/
├── results_small_100_50_hybrid.json  # Baseline 结果
└── results_small_100_50_rerank.json  # Rerank 结果
```

### 结果文件结构

```json
{
  "config": {
    "chunking_strategy": "small_100_50",
    "retrieval_mode": "rerank",
    "k": 5
  },
  "scores": { "overall": 0.XX, ... },
  "mrr_scores": { "overall": 0.XX, ... },
  "rerank_score_distribution": {
    "statistics": { ... },
    "threshold_suggestion": { ... }
  },
  "results": [
    {
      "id": 1,
      "query": "...",
      "hit": 1,
      "answer_rank": 1,
      "rerank_scores": [0.85, 0.72, 0.65, 0.58, 0.42],
      "answer": "..."
    },
    ...
  ]
}
```

---

## 🔧 常见问题

### Q1: FlagEmbedding 安装失败

```bash
# 尝试指定版本
pip install FlagEmbedding==1.2.10

# 或者使用清华镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple FlagEmbedding
```

### Q2: 模型下载慢或失败

**方法1：使用 HuggingFace 镜像**
```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 test_reranker.py
```

**方法2：手动下载模型**
```bash
# 从 https://hf-mirror.com/BAAI/bge-reranker-base 下载
# 放到 ~/.cache/huggingface/hub/ 目录
```

### Q3: CUDA/GPU 相关错误

Reranker 默认使用 CPU 也能运行（会慢一些）。如果出现 CUDA 错误：

```python
# 在 retrieval.py 中修改：
_reranker_model = FlagReranker(
    'BAAI/bge-reranker-base',
    use_fp16=False,  # 改为 False，使用 FP32
    device='cpu'     # 强制使用 CPU
)
```

### Q4: 内存不足

如果评估时内存不足，可以减少候选数量：

```bash
# 在 retrieval.py 的 retrieve_rerank() 中
# 将 k_candidates=20 改为 k_candidates=10
```

---

## 📈 验收标准

### 最低目标（必须达成）

| 指标 | Baseline (Hybrid) | Rerank 目标 | 提升 |
|------|------------------|------------|------|
| Overall Recall@5 | 0.78 | ≥ 0.90 | +15% |
| Overall MRR | 0.57 | ≥ 0.78 | +37% |
| chunking_sens Recall | 0.60 | ≥ 0.80 | +33% |

### 理想目标（争取达成）

| 指标 | Baseline (Hybrid) | Rerank 目标 | 提升 |
|------|------------------|------------|------|
| Overall Recall@5 | 0.78 | ≥ 0.94 | +21% |
| Overall MRR | 0.57 | ≥ 0.85 | +49% |
| chunking_sens MRR | 0.38 | ≥ 0.70 | +84% |

---

## 🎯 下一步

运行完评估后，创建对比分析：

```bash
python3 compare_results.py      # Step 6: 结果对比
python3 analyze_rerank_scores.py  # Step 7: 分数分析
```

然后更新文档：
- README.md
- Iteration4.md（验收报告）
