# Iteration 4: Reranker（精排优化）

## 🎯 迭代目标

实现两阶段检索架构：Hybrid Search（粗排）+ Reranker（精排），提升检索精度和排名质量。

## 📊 性能表现

**配置：** `fixed_200_40` + `hybrid` (召回 top-20) + `bge-reranker-base` (精排 top-5)

| 指标 | Baseline (Hybrid) | Rerank | 提升 |
|------|-------------------|--------|------|
| **Overall Recall@5** | 0.69 | **0.94** | **+36%** 🔥 |
| **Overall MRR** | 0.5547 | **0.7604** | **+37%** 🔥 |
| chunking_sensitive Recall | 0.30 | **0.90** | **+200%** 🔥 |
| exact_match Recall | 1.00 | 1.00 | 持平 ✅ |
| semantic_paraphrase Recall | 0.73 | **0.91** | **+25%** ✅ |

**核心成果：**
- ✅ Overall Recall 达到 0.94（超过理想目标）
- ✅ chunking_sensitive 提升惊人（+200%）
- ✅ 所有类别都有提升或持平

---

## 🏗️ 技术架构

### 两阶段检索流程

```
User Query
    ↓
┌─────────────────────────────────┐
│  Hybrid Search (粗排)            │
│  - Vector Search (语义匹配)      │
│  - BM25 (关键词匹配)             │
│  - RRF 融合                      │
│  → 召回 Top-20 候选              │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Reranker (精排)                 │
│  - bge-reranker-base             │
│  - Cross-encoder 架构            │
│  - 精确打分                      │
│  → 精排 Top-5 结果               │
└─────────────────────────────────┘
    ↓
LLM 生成答案
```

### Cross-encoder vs Bi-encoder

| 特性 | Bi-encoder (Vector Search) | Cross-encoder (Reranker) |
|------|---------------------------|--------------------------|
| **编码方式** | 分别编码 query 和 doc | 同时编码 query + doc |
| **相似度计算** | 余弦相似度 | 直接预测相关性 |
| **精度** | 较低 | 高 ✅ |
| **速度** | 快（预计算向量）✅ | 慢（实时计算） |
| **适用阶段** | 召回（粗排） | 精排 |

**为什么需要两阶段？**
- 候选文档多时，Cross-encoder 计算成本高（N 次推理）
- 先用快速方法召回候选，再用精确方法精排
- 平衡速度和精度

---

## 📁 文件结构

```
iteration4/
├── corpus/                 # 测试语料（15个文档）
│   ├── doc-1.txt ~ doc-7.txt   # 原始文档
│   ├── doc-8.txt ~ doc-15.txt  # 干扰文档
│   └── queries.json            # 32个测试查询
│
├── chunking.py            # 分块策略（继承自 Iteration 2/3）
├── retrieval.py           # 检索模块（新增 retrieve_rerank）
├── generation.py          # 生成模块（继承自 Iteration 0）
├── scoring.py             # 评分模块（新增 MRR 和分数分析）
├── run_eval.py            # 评估脚本（支持 rerank 模式）
│
├── test_reranker.py       # Reranker 功能测试
├── compare_results.py     # Hybrid vs Rerank 对比分析
├── analyze_rerank_scores.py  # 分数分布和阈值分析
│
├── requirements.txt       # 依赖（新增 FlagEmbedding）
├── RUN_INSTRUCTIONS.md    # 运行说明
├── Iteration4.md          # 验收报告
└── 干扰文档说明.md         # 测试集设计说明
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd iteration4
pip install -r requirements.txt
```

**新增依赖：**
- `FlagEmbedding>=1.2.0` - bge-reranker-base 模型

### 2. 测试 Reranker

```bash
python3 test_reranker.py
```

**预期输出：**
```
✅ FlagEmbedding 导入成功
✅ Reranker 模型加载成功
✅ 分数范围正常 (0-1)
✅ 排序逻辑正确
🎉 所有测试通过！
```

### 3. 运行评估

```bash
# Baseline (Hybrid Search)
python3 run_eval.py --chunking-strategy fixed_200_40 --retrieval-mode hybrid

# Rerank (Hybrid + Reranker)
python3 run_eval.py --chunking-strategy fixed_200_40 --retrieval-mode rerank
```

### 4. 对比分析

```bash
# 对比 Hybrid vs Rerank
python3 compare_results.py

# 分析 Rerank 分数分布
python3 analyze_rerank_scores.py
```

---

## 🔑 核心实现

### retrieve_rerank() 函数

```python
def retrieve_rerank(
    query: str,
    chunks: List[Dict],
    k: int = 5,
    strategy: str = "fixed_200_40",
    k_candidates: int = 20
) -> List[Dict]:
    """两阶段检索：Hybrid 召回 + Reranker 精排"""
    
    # Step 1: Hybrid Search 召回 top-20
    candidates = retrieve_hybrid(
        query=query,
        chunks=chunks,
        k=k_candidates,
        strategy=strategy
    )
    
    # Step 2: Reranker 精排
    reranker = _get_reranker_model()
    pairs = [[query, c['text']] for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    
    # Step 3: 添加分数并排序
    for i, candidate in enumerate(candidates):
        candidate['rerank_score'] = float(scores[i])
    
    reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
    
    return reranked[:k]
```

### Reranker 模型

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker(
    'BAAI/bge-reranker-base',
    use_fp16=True  # 使用半精度加速
)

# 计算相关性分数
scores = reranker.compute_score(
    [[query, doc1], [query, doc2], ...],
    normalize=True  # 归一化到 [0, 1]
)
```

---

## 📊 测试集设计

### 干扰文档策略

为了有效测试 Reranker，新增了 8 个干扰文档：

1. **相似产品文档（5个）**
   - SmartLock-200, SmartCam-300, SmartPlug-500 等
   - 功能重叠但型号不同
   - 测试 Reranker 的型号识别能力

2. **通用安装指南（1个）**
   - 包含所有型号，但只讲安装步骤
   - 测试信息完整性判断

3. **用户FAQ（1个）**
   - 口语化、碎片化信息
   - 测试权威性和完整性判断

4. **行业新闻（1个）**
   - 营销文案，关键词密度高
   - 测试实质内容 vs 宣传文案的区分

**效果：**
- 文档数：7 → 15 (+114%)
- Baseline Recall 下降：0.97 → 0.69 (-29%)
- 为 Reranker 提供了真实挑战

---

## 🎓 关键发现

### 1. Reranker 对不同查询类型的影响

**chunking_sensitive（+200% Recall）:**
- 这类查询最依赖精确的文档块选择
- Cross-encoder 能理解细粒度的语义匹配
- 示例：ST-500 传感器安装高度、防猫眼功能操作

**exact_match（保持 1.00）:**
- 得益于 Hybrid 的 BM25 关键词匹配
- Reranker 进一步优化排名（MRR +11%）
- 示例：产品型号、参数查询

**semantic_paraphrase（+25% Recall）:**
- Reranker 能理解语义等价
- 示例："晚上能看多远" = "夜视距离"

### 2. 分数分布特征

- **命中查询 top-1 均分：** 0.8935
- **未命中查询 top-1 均分：** 0.9057
- **高度重叠：** 命中和未命中的分数区分度不够
- **原因：** 测试集太简单（94% 命中率）

### 3. 阈值建议（为 Iteration 6 准备）

| 阈值 | 策略 | 拒答率 | 误拒率 |
|------|------|--------|--------|
| 0.95 | 保守 | 低 | 低 |
| 0.85-0.90 | **推荐** | 中 | 中 |
| 0.70-0.80 | 激进 | 高 | 高 |

**注意：** 当前数据集只有 6% 的无法回答查询，真实场景需要更多无答案测试用例。

---

## ⚠️ 已知限制

1. **MRR 未达理想目标**
   - 当前 0.7604，理想目标 0.85
   - 部分查询答案排名不够靠前（不在 top-1）

2. **个别 exact_match 查询下降**
   - Query 18: SmartPlug-400 待机功耗
   - Hybrid 命中但 Rerank 未命中
   - 可能需要针对 exact_match 的混合策略

3. **阈值机制需要更真实数据**
   - 当前无答案查询太少（6%）
   - 无法有效验证拒答阈值
   - Iteration 6 需要扩展测试集

---

## 🔄 与其他迭代的关系

### 继承自前序迭代

- **Iteration 0:** Generation 模块
- **Iteration 1:** Vector Search（用于 Hybrid）
- **Iteration 2:** Chunking 策略（fixed_200_40）
- **Iteration 3:** Hybrid Search（召回阶段）

### 为后续迭代准备

- **Iteration 5 (Faithfulness):**
  - Reranker 提升了检索质量
  - 应该能改善生成答案的忠实度

- **Iteration 6 (拒答机制):**
  - Rerank 分数可作为置信度指标
  - 低分数（<0.85）可触发拒答
  - 需要更多"无法回答"的测试用例

---

## 📚 相关文档

- [Iteration4.md](./Iteration4.md) - 完整验收报告
- [RUN_INSTRUCTIONS.md](./RUN_INSTRUCTIONS.md) - 详细运行说明
- [干扰文档说明.md](./干扰文档说明.md) - 测试集设计文档
- [docs/iteration-plan.md](../docs/iteration-plan.md) - 整体迭代计划

---

## 🎯 总结

Iteration 4 成功实现了 Reranker 并验证了两阶段检索架构的有效性：

✅ **核心成果：**
- Overall Recall 0.94（+36%）
- Overall MRR 0.76（+37%）
- chunking_sensitive 提升惊人（+200%）

✅ **技术价值：**
- Cross-encoder 精度显著高于 Bi-encoder
- 两阶段架构平衡了速度和精度
- 分数分布为拒答机制提供了基础

⚠️ **改进空间：**
- MRR 可进一步优化（当前 0.76，目标 0.85）
- 需要更真实的测试集（更多无答案查询）
- 可考虑针对查询类型的混合策略

**下一步：** 进入 Iteration 5，实现 Faithfulness 评估，检查生成答案是否忠实于检索到的文档。
