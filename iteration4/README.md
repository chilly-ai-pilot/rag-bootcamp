# Iteration 4: Reranker（准备阶段）

## 概述

Iteration 4 将在 Iteration 3 的混合检索基础上，引入 **Reranker（重排序器）** 来优化检索结果的排序质量。本次先建立 **baseline（基线）**，引入 **MRR (Mean Reciprocal Rank)** 指标来衡量答案块的排名质量。

**目标：** 
1. 建立带 MRR 指标的 baseline
2. 为 Reranker 的效果评估做准备
3. 增加干扰文档，制造更多困难案例

**当前状态：** Baseline 已建立，使用 `fixed_200_40 + hybrid` 配置

---

## 新增功能

### 1. MRR (Mean Reciprocal Rank) 指标

**什么是 MRR？**

MRR 衡量答案块在检索结果中的排名质量。Recall@K 只关心"答案是否在 top-K 内"，但不关心答案排在第 1 还是第 5。MRR 关注答案的具体排名：

```python
# 答案排第1名：1/1 = 1.0  (完美)
# 答案排第3名：1/3 = 0.33 (中等)
# 答案排第5名：1/5 = 0.20 (较差)
# 未命中：0.0
```

**为什么需要 MRR？**

Reranker 的核心价值是 **把答案往前推**。即使 Recall@5 不变，如果 MRR 提升，说明：
- 答案块排名更靠前
- 生成器更容易"看到"正确答案
- 生成质量更高

**对比示例：**

| Query | Vector 排名 | Rerank 排名 | Recall@5 | MRR 变化 |
|-------|------------|------------|----------|---------|
| ID 1 | 答案第3位 | 答案第1位 | 无变化 (都命中) | 0.33→1.0 (+0.67) ✅ |
| ID 9 | 答案第3位，噪声第5位 | 答案第1位，噪声降到第8位 | 无变化 | 0.33→1.0 (+0.67) ✅ |

### 2. 新增代码

**`scoring.py` 新增函数：**
```python
def calculate_mrr(results: List[Dict]) -> Dict[str, float]:
    """计算 MRR (Mean Reciprocal Rank)"""
    
def find_answer_rank(retrieved_chunks: List[Dict], gt_doc_id: str, 
                     gt_start: int, gt_end: int) -> int:
    """找到答案块在检索结果中的排名（1-based）"""
```

**`run_eval.py` 修改：**
- 为每个 query 记录 `answer_rank`（答案块排名）
- 计算并显示 MRR 分数
- 结果文件中保存 MRR 数据

---

## Baseline 结果

### 配置

| 组件 | 选择 | 说明 |
|------|------|------|
| **Chunking** | fixed_200_40 | 长块策略，块数少（28块） |
| **Retrieval** | hybrid (RRF) | 向量 + BM25 融合 |
| **召回数量** | k_vector=20, k_bm25=20 | 最终返回 top-5 |

### 性能指标

**Recall@5：**

| 类别 | 分数 |
|------|------|
| chunking_sensitive | 0.90 |
| exact_match | 1.00 |
| semantic_paraphrase | 1.00 |
| **overall** | **0.97** |

**MRR (Mean Reciprocal Rank)：**

| 类别 | 分数 |
|------|------|
| chunking_sensitive | 0.5000 |
| exact_match | 1.0000 |
| semantic_paraphrase | 1.0000 |
| **overall** | **0.8438** |

### 关键洞察

1. **exact_match 的 MRR 是满分 (1.0)**
   - 说明产品型号、编号等精确匹配的 query，答案块都排在第 1 位
   - 这类 query 已经很好，Reranker 提升空间有限

2. **chunking_sensitive 的 MRR 只有 0.5**
   - 说明这类 query 的答案块平均排在第 2-3 位
   - **这是 Reranker 的主要优化目标**
   - 目标：MRR 从 0.5 提升到 0.7-0.8

3. **overall MRR = 0.8438**
   - 这是一个相对较高的基线
   - Reranker 的目标：提升到 0.90+

---

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

（与 Iteration 3 相同）

### 运行 Baseline 评估

```bash
# 使用推荐的 baseline 配置
python3 run_eval.py --chunking-strategy fixed_200_40 --retrieval-mode hybrid
```

### 查看结果

```bash
cat results_fixed_200_40_hybrid.json
```

结果文件包含：
- `config`: 配置信息
- `scores`: Recall@K 分数
- `mrr_scores`: MRR 分数（新增）
- `results`: 每个 query 的详细结果（包含 `answer_rank`）

---

## 下一步计划

### Phase 1: 增加干扰文档（准备中）

**目标：** 增加 10-15 个干扰文档，让 baseline 更具挑战性

**新增文档类型：**
- 5-10 个相似产品文档（功能重叠的产品）
- 5-10 个干扰文档（同领域但不相关的内容）
- 目标文档数：20-30 篇

**预期效果：**
- Recall@5 可能下降到 0.90-0.93
- MRR 可能下降到 0.75-0.80
- 制造更多"看起来相关但不相关"的候选块
- Reranker 的去噪能力更容易体现

### Phase 2: 实现 Reranker

**技术方案：**
```
Vector top-20  ─┐
                ├─→ RRF 融合 → top-20 候选
BM25 top-20   ─┘
                    ↓
              Reranker (bge-reranker-base)
           重新评估 query-doc 匹配度
                    ↓
                 top-5（精排后）
```

**验收目标：**
- Recall@5: 保持或提升（≥ baseline）
- MRR: 提升 5-10%（例如 0.84 → 0.90+）
- 至少 50% 的 query 答案排名提升

---

## 文件结构

```
iteration4/
├── corpus/                    # 语料库（待扩充）
│   ├── doc-1.txt ~ doc-7.txt  # 当前7个文档
│   └── queries.json           # 32 条测试查询
├── chunking.py                # Chunking 策略
├── retrieval.py               # 检索模块（vector, bm25, hybrid）
├── generation.py              # 答案生成
├── scoring.py                 # ✨ 新增：MRR 计算
├── run_eval.py                # ✨ 更新：支持 MRR
├── requirements.txt           # 依赖
├── results_fixed_200_40_hybrid.json  # Baseline 结果
└── README.md                  # 本文件
```

---

## 主要代码变更

### `scoring.py` 新增

```python
def calculate_mrr(results: List[Dict]) -> Dict[str, float]:
    """计算 MRR (Mean Reciprocal Rank)"""
    # 按类别计算平均倒数排名
    # 答案排第1名：1/1 = 1.0
    # 答案排第3名：1/3 = 0.33
    # 未命中：0.0

def find_answer_rank(retrieved_chunks: List[Dict], 
                     gt_doc_id: str, gt_start: int, gt_end: int) -> int:
    """找到答案块在检索结果中的排名（1-based）"""
    # 返回答案块的排名，如果未找到返回 None
```

### `run_eval.py` 修改

```python
# 1. 记录答案块排名
answer_rank = find_answer_rank(retrieved, q["doc_id"], q["char_start"], q["char_end"])
results.append({
    "id": q["id"],
    "query": q["query"],
    "category": q["category"],
    "hit": h,
    "answer_rank": answer_rank,  # 新增
    "answer": answer,
})

# 2. 计算 MRR
mrr_scores = calculate_mrr(results)

# 3. 显示 MRR
print(f"\n=== MRR (Mean Reciprocal Rank) ===")
for cat, score in mrr_scores.items():
    print(f"  {cat:24s} {score:.4f}")
```

---

## 验收状态

| 目标 | 状态 | 说明 |
|------|------|------|
| 建立 baseline | ✅ 完成 | fixed_200_40 + hybrid |
| 实现 MRR 指标 | ✅ 完成 | scoring.py 和 run_eval.py |
| 运行 baseline 评估 | ✅ 完成 | overall Recall=0.97, MRR=0.8438 |
| 增加干扰文档 | ⏳ 待进行 | 目标：20-30 篇文档 |
| 实现 Reranker | ⏳ 待进行 | 下一阶段 |

---

## 常见问题

### Q: 为什么选择 fixed_200_40 + hybrid 作为 baseline？

**A:** 基于 Iteration 3 的发现：
1. 达到 0.97 的性能（与 small_100_50 + vector 并列最优）
2. 块数少（28 vs 85），扩展到更多文档时性能更好
3. Hybrid 会引入 BM25 噪声，Reranker 可以展示去噪能力
4. 这是工业界的标准做法（hybrid → rerank）

### Q: MRR 为什么不是 1.0？

**A:** MRR = 0.8438 说明：
- 有些 query 的答案块不在第 1 位（排在第 2、3 位）
- 特别是 chunking_sensitive 类别，MRR 只有 0.5（平均排第 2 位）
- 这正是 Reranker 的优化空间

### Q: 为什么要增加干扰文档？

**A:** 当前 7 个文档太"干净"：
- 检索器很容易区分相关和不相关的块
- Reranker 的去噪能力看不出来
- 增加干扰文档后，会有更多"看起来相关但不相关"的候选块
- Reranker 的价值才能充分体现

### Q: Baseline 已经 0.97 了，Reranker 还有用吗？

**A:** 有用！Reranker 的价值不只是提升 Recall@K：
1. **提升 MRR**：把答案从第 2-3 位推到第 1 位
2. **去噪**：过滤"看起来相关但不相关"的块
3. **更高的生成质量**：答案越靠前，生成器越容易使用
4. **为拒答机制铺路**：Reranker 分数可用于设置拒答阈值

---

## 参考资料

- **MRR 解释**: [Mean Reciprocal Rank - Wikipedia](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
- **Reranker 原理**: Cross-encoder vs Bi-encoder
- **bge-reranker**: https://huggingface.co/BAAI/bge-reranker-base

---

## 总结

Iteration 4 的 baseline 已建立：

**✅ 已完成：**
- 实现 MRR 指标
- 建立 baseline（fixed_200_40 + hybrid）
- 运行评估并记录性能

**🎯 当前性能：**
- Recall@5: 0.97
- MRR: 0.8438
- chunking_sensitive MRR: 0.5（主要优化目标）

**🚀 下一步：**
1. 增加 10-15 个干扰文档
2. 实现 bge-reranker-base
3. 对比 baseline vs rerank
4. 目标：MRR 提升到 0.90+

**核心价值：**

MRR 指标让我们能够精确衡量"答案块排名质量"，这是 Reranker 的核心优化目标。即使 Recall@K 不变，MRR 的提升也说明系统在变得更好。
