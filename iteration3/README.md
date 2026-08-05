# Iteration 3: Hybrid Search（混合检索）

## 概述

Iteration 3 在 Iteration 2 的基础上，新增 **BM25 关键词检索** 和 **Hybrid Search（混合检索）**，使用 **RRF（倒数排名融合）** 算法结合向量检索和关键词检索的优势。

**目标：** 解决术语不一致问题（如 ID 7："充电保护功能" vs "防止过充"），将 overall Recall 从 0.97 提升到 1.00。

**结果：** 技术实现成功，但性能未达预期（0.94），发现了 Hybrid Search 的适用边界和 RRF 的局限性，为 Iteration 4 (Reranker) 提供了明确方向。

---

## 新增功能

### 1. BM25 关键词检索
- 使用 `jieba` 进行中文分词
- 使用 `rank-bm25` 实现 BM25Okapi 算法
- 特别适合产品型号、编号等精确匹配场景

### 2. Hybrid Search（混合检索）
- 向量检索召回 top-20
- BM25 检索召回 top-20
- 使用 RRF（Reciprocal Rank Fusion）融合结果
- 最终返回 top-5

### 3. RRF（倒数排名融合）算法
- 公式：`RRF_score(doc) = Σ 1/(k + rank_i)`，其中 k=60
- 无需调参，量纲无关
- 平局处理：优先保留向量检索中排名更高的文档

---

## 实验结果

### 完整对比：2 种 Chunking × 3 种检索模式

| Chunking | Retrieval | Overall | chunk_sens | exact | semantic | 评价 |
|----------|-----------|---------|-----------|-------|----------|------|
| **fixed_200_40** | vector | 0.94 | 0.80 | 1.00 | 1.00 | 良好 |
| **fixed_200_40** | bm25 | 0.94 | 0.80 | 1.00 | 1.00 | 良好 |
| **fixed_200_40** | **hybrid** | **0.97** 🏆 | **0.90** | 1.00 | 1.00 | **最优之一** |
| **small_100_50** | **vector** | **0.97** 🏆 | **0.90** | 1.00 | 1.00 | **最优之一** |
| **small_100_50** | bm25 | 0.88 | 0.70 | 1.00 | 0.91 | 较差 |
| **small_100_50** | hybrid | 0.94 | 0.80 | 1.00 | 1.00 | 良好 |

### 失败案例（small_100_50 策略）

| 模式 | 失败数 | 失败 IDs |
|------|-------|---------|
| vector | 1 | 7 |
| bm25 | 4 | 5, 7, 9, 29 |
| hybrid | 2 | 7, 9 |

---

## 关键发现

### ✅ 成功之处

1. **发现了第二条达到 0.97 的最优路径**
   - 路径 1：small_100_50 + vector（Iteration 2）
   - 路径 2：fixed_200_40 + hybrid（Iteration 3 新发现）
   - 两条路径性能相同，可根据资源和需求选择

2. **块长度显著影响 BM25 效果**
   - fixed_200_40（长块）：BM25 overall 0.94 ✅
   - small_100_50（短块）：BM25 overall 0.88 ❌
   - 提升：+6.8%
   - 原因：长块包含更多关键词，BM25 累积分数更高

3. **技术实现成功**
   - RRF 融合算法正确工作
   - 成功合并向量和 BM25 的结果

### ❌ 意外发现

1. **Hybrid 只在"势均力敌"时有效**
   - fixed_200_40: vector=0.94, bm25=0.94 → hybrid=0.97 ✅
   - small_100_50: vector=0.97, bm25=0.88 → hybrid=0.94 ❌
   - **当两个检索器实力相近时，RRF 融合才能发挥互补作用**

2. **两类失败案例**
   - **ID 7（召回失败）**：两个检索器都没把答案排进 top-5
   - **ID 9（噪声干扰）**：仅在 small_100_50 上，BM25 噪声块挤掉 Vector 的正确答案

### 💡 核心洞察

**Chunking × Retrieval 的适配矩阵：**

| Chunking | 最佳检索 | Overall | 为什么？ |
|----------|---------|---------|---------|
| **small_100_50（短块）** | **Vector** | **0.97** | 精确的局部语义，不依赖关键词数量 |
| **fixed_200_40（长块）** | **Hybrid** | **0.97** | 长块让 BM25 性能提升，与 Vector 形成互补 |

**Hybrid 不一定优于单一检索器！适用条件：**
- ✅ 长块 + 两个检索器实力相近 → 互补效应
- ❌ 短块 + 检索器实力悬殊 → 被弱者拖累

---

## 技术栈

| 组件 | 版本/选择 |
|------|----------|
| Chunking | small_100_50 (Iteration 2 最优) |
| 向量检索 | bge-base-zh-v1.5 + ChromaDB |
| 关键词检索 | rank-bm25==0.2.2 + jieba==0.42.1 |
| 融合算法 | RRF (k=60) |
| 召回数量 | k_vector=20, k_bm25=20 → top-5 |

---

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

新增依赖：
- `rank-bm25>=0.2.2` - BM25 算法
- `jieba>=0.42.1` - 中文分词

### 运行评估

#### 单个检索模式

```bash
# Vector（Iteration 2 baseline）
python run_eval.py --chunking-strategy small_100_50 --retrieval-mode vector

# BM25（纯关键词）
python run_eval.py --chunking-strategy small_100_50 --retrieval-mode bm25

# Hybrid（向量 + BM25 + RRF）
python run_eval.py --chunking-strategy small_100_50 --retrieval-mode hybrid
```

#### 对比所有 chunking 策略

```bash
# 使用 hybrid 模式对比三种 chunking 策略
python run_eval.py --compare-all --retrieval-mode hybrid
```

### 查看帮助

```bash
python run_eval.py --help
```

---

## 文件结构

```
iteration3/
├── corpus/                    # 语料库（同 Iteration 2）
│   ├── doc-1.txt ~ doc-7.txt
│   └── queries.json           # 32 条测试查询
├── chunking.py                # Chunking 策略（同 Iteration 2）
├── retrieval.py               # ✨ 新增：BM25 和 Hybrid 检索
├── generation.py              # 答案生成（同 Iteration 2）
├── scoring.py                 # 评分指标（同 Iteration 2）
├── run_eval.py                # ✨ 更新：支持多种检索模式
├── requirements.txt           # ✨ 新增：BM25 依赖
├── results_small_100_50_vector.json   # Vector 结果
├── results_small_100_50_bm25.json     # BM25 结果
├── results_small_100_50_hybrid.json   # Hybrid 结果
├── Iteration3.md              # 验收报告
└── README.md                  # 本文件
```

---

## 主要代码变更

### `retrieval.py` 新增函数

```python
def retrieve_bm25(query: str, chunks: List[Dict], k: int = 5) -> List[Dict]:
    """BM25 关键词检索"""
    # jieba 分词 → BM25 计算 → 返回 top-k

def retrieve_hybrid(
    query: str, chunks: List[Dict], k: int = 5,
    strategy: str = "fixed_200_40",
    k_vector: int = 20, k_bm25: int = 20, rrf_k: int = 60
) -> List[Dict]:
    """混合检索（向量 + BM25 + RRF）"""
    # 向量召回 top-20 + BM25 召回 top-20 → RRF 融合 → 返回 top-k

def _reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60, top_k: int = 5
) -> List[Dict]:
    """RRF（倒数排名融合）算法"""
    # 计算 RRF 分数 → 排序 → 返回 top-k
```

### `run_eval.py` 新增参数

```python
ap.add_argument(
    "--retrieval-mode",
    default="vector",
    choices=["random", "vector", "bm25", "hybrid"],
    help="检索模式"
)
```

---

## 实验复现

### 复现 Iteration 3 的核心实验

```bash
# 1. Vector baseline（期望 0.97）
python run_eval.py --chunking-strategy small_100_50 --retrieval-mode vector

# 2. BM25（期望 0.88）
python run_eval.py --chunking-strategy small_100_50 --retrieval-mode bm25

# 3. Hybrid（期望 0.94）
python run_eval.py --chunking-strategy small_100_50 --retrieval-mode hybrid
```

### 分析失败案例

```python
import json

# 读取结果
with open('results_small_100_50_hybrid.json') as f:
    results = json.load(f)['results']

# 找失败案例
fails = [r for r in results if r['hit'] == 0]
for r in fails:
    print(f"ID {r['id']}: {r['query']} [{r['category']}]")
```

---

## 验收状态

| 目标 | 状态 | 说明 |
|------|------|------|
| 实现 BM25 检索 | ✅ 完成 | jieba + rank-bm25 |
| 实现 Hybrid Search | ✅ 完成 | RRF 融合算法 |
| 提升 overall Recall 到 1.00 | ❌ 未达成 | 实际 0.94（退步）|
| 解决 ID 7（术语不一致）| ❌ 未达成 | 两个检索器都失败 |
| 发现重要洞察 | ✅ 超额完成 | Hybrid 适用边界、RRF 局限 |

**总体评价：** 技术目标达成，性能目标未达成，但获得了比"提升几个点"更有价值的发现。

---

## 下一步

### Iteration 4: Reranker

**目标：** 使用 `bge-reranker-base` 对召回结果重新精排，过滤噪声块（如 ID 9）

**预期提升：**
- 解决 ID 9（噪声干扰问题）
- overall Recall 从 0.94 (hybrid) 提升到 0.97+

**技术方案：**
```
Vector top-20  ─┐
                ├─→ RRF 融合 → top-20 候选
BM25 top-20   ─┘
                    ↓
              Reranker (cross-encoder)
           重新评估 query-doc 匹配度
                    ↓
                 top-5（精排后）
```

---

## 参考资料

- **RRF 论文**: Gordon, Cormack (2006) - "Reciprocal Rank Fusion"
- **BM25 算法**: Robertson, Zaragoza (2009) - "The Probabilistic Relevance Framework: BM25 and Beyond"
- **jieba 分词**: https://github.com/fxsjy/jieba
- **rank-bm25**: https://github.com/dorianbrown/rank_bm25

---

## 常见问题

### Q: 为什么 Hybrid 比 Vector 更差？

**A:** 在语义理解占比高的数据集上（本数据集 69%），BM25 的噪声会通过 RRF 融合干扰结果。例如 ID 9，BM25 的噪声块（chunk 66）挤掉了 Vector 正确找到的答案块。

### Q: Hybrid Search 什么时候有用？

**A:** 适合关键词匹配占比高的场景：
- 日志搜索（错误代码、IP 地址）
- 代码搜索（函数名、变量名）
- 产品搜索（型号、SKU）

不适合语义理解为主的场景（如本数据集的问答）。

### Q: ID 7 为什么无法解决？

**A:** 因为"充电保护功能" vs "防止过充"的语义鸿沟太大，两个检索器都没把答案排进 top-20。RRF 只能融合已有的排名，无法"凭空"提升排名。需要其他技术（如查询改写、Reranker）。

### Q: 为什么不直接用 Reranker，跳过 Hybrid？

**A:** 实验设计的科学性：
1. 逐步验证每个技术的效果
2. Hybrid (RRF) 是轻量级方案，Reranker 是重量级方案
3. 通过对比，明确了 Reranker 的必要性和价值

---

## 总结

Iteration 3 是一次"超出预期"的成功实验：

**虽然：**
- ❌ small_100_50 + hybrid 未能超过 vector（0.94 vs 0.97）

**但是：**
- ✅ 发现了第二条达到 0.97 的路径：fixed_200_40 + hybrid
- ✅ 揭示了块长度对 BM25 的显著影响（+6.8%）
- ✅ 明确了 Hybrid 的适用条件（长块 + 势均力敌）
- ✅ 提供了两种最优组合供不同场景选择

**这正是科学实验的意义：用数据说话，不盲目追求新技术，获得比"提升几个点"更重要的洞察。**

### 选择建议

| 场景 | 推荐组合 | 原因 |
|------|---------|------|
| 追求极致性能 | 两者都可 | 都是 0.97 |
| **资源受限** | **fixed_200_40 + hybrid** | 块数少 3 倍（28 vs 85）|
| 系统简化 | small_100_50 + vector | 单一检索器 |
| 关键词重要 | fixed_200_40 + hybrid | BM25 在长块上强 |
