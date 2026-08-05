# Iteration 2: Chunking 策略优化

**目标**：对比三种不同的文档切块方法，找到最适合当前语料的切块策略，解决 Iteration 1 中发现的"切块边界切断答案"问题。

**验收标准**：用相同的测试集分别测试三种 chunking 策略，对比 Recall@K 的变化，重点关注 chunking_sensitive 类别的提升（从 0.80 提升到 0.90+）。

---

## 核心改进

相比 Iteration 1，本次迭代的关键变化：

| 组件 | Iteration 1 | Iteration 2 | 说明 |
|------|-------------|-------------|------|
| Chunking 策略 | 单一策略（fixed_200_40） | **三种策略对比** | 系统化实验，找到最优策略 |
| `chunking.py` | 单一函数 | 支持策略切换 | 新增 `semantic` 和 `small_100_50` 策略 |
| `retrieval.py` | 单一 collection | **多 collection 支持** | 每个策略使用独立的向量数据库 |
| `run_eval.py` | 单次运行 | **批量对比模式** | `--compare-all` 一键运行所有策略 |

**保持不变**：
- Embedding 模型（bge-base-zh-v1.5）
- 向量数据库（ChromaDB）
- 生成器（DeepSeek API）
- 评分指标（Recall@K）

---

## 三种 Chunking 策略详解

### 1. fixed_200_40（Iteration 1 baseline）

**策略**：固定长度切块，200 字符，40 字符重叠

**优势**：
- 简单、可控、切块数量适中
- 块大小均匀，便于向量检索

**劣势**：
- 可能在句子中间切断，破坏语义完整性
- Iteration 1 发现：ID 3 案例中，答案被切在块边界导致未命中

**适用场景**：文档结构简单、句子较短的场景

---

### 2. semantic（按句子边界切分）

**策略**：按句子标点（句号、问号、感叹号）切分，累积到目标大小（~200字符），不超过最大值（350字符）

**优势**：
- **保持语义完整性**：每个块都是完整的句子，不会在句子中间切断
- 解决 Iteration 1 的 ID 3 问题：答案不会被切断

**劣势**：
- 块大小不均匀（可能影响检索一致性）
- 对没有标点的文本效果不佳（会回退到固定长度）

**适用场景**：有明确句子边界的结构化文本（如产品文档、FAQ）

**实现细节**：
```python
# 示例：一个长句子（150字符）+ 一个短句子（80字符）= 一个块（230字符）
# 如果再加一个句子会超过 350 字符，就保存当前块，开始新块
```

---

### 3. small_100_50（更小粒度 + 更大重叠）

**策略**：100 字符，50 字符重叠（50% 重叠率）

**优势**：
- 更细粒度：更精确地定位到答案
- **更大重叠率**（50% vs 20%）：大幅减少边界切断问题

**劣势**：
- 块数量增加约 2 倍：索引和检索开销增大
- 可能导致冗余信息过多

**适用场景**：对精确度要求极高、愿意牺牲性能的场景

---

## 技术选型说明

### 为什么需要对比 Chunking 策略？

Chunking 是 RAG 系统中**被低估的杠杆**：
- 如果切块边界不合理，再好的 embedding 模型也无法检索到正确答案
- Iteration 1 的实验证实：2/32 的未命中案例中，至少 1 个（ID 3）是由切块边界问题导致的
- 迭代计划明确指出：**这一步应该是整个项目里，你能看到分数提升最明显的一次迭代**

---

## 文件结构

```
iteration2/
├── requirements.txt      复用 iteration1（未改动）
├── chunking.py           核心改动：支持三种策略（fixed_200_40, semantic, small_100_50）
├── retrieval.py          核心改动：支持多 collection（每个策略独立）
├── generation.py         复用 iteration1（未改动）
├── scoring.py            复用 iteration1（未改动）
├── run_eval.py           核心改动：新增 --compare-all 批量对比模式
├── corpus/               复用 iteration0 的测试集
├── chroma_db/            自动生成：存储多个 collection（rag_docs_fixed_200_40, rag_docs_semantic, rag_docs_small_100_50）
├── results_fixed_200_40.json   自动生成：策略1的结果
├── results_semantic.json       自动生成：策略2的结果
├── results_small_100_50.json   自动生成：策略3的结果
├── Iteration2.md         本次迭代的验收报告（运行实验后填写）
└── README.md             本文件
```

---

## 使用方法

### 1. 安装依赖

```bash
cd iteration2
pip install -r requirements.txt
```

依赖与 Iteration 1 相同，如果已经安装过，无需重新安装。

---

### 2. 运行单个策略

```bash
# 策略 1：fixed_200_40（Iteration 1 baseline）
python run_eval.py --chunking-strategy fixed_200_40

# 策略 2：semantic（按句子边界切分）
python run_eval.py --chunking-strategy semantic

# 策略 3：small_100_50（更小粒度+更大重叠）
python run_eval.py --chunking-strategy small_100_50
```

**输出**：
- 终端显示：块数量统计、前 3 个示例、Recall@K 分数
- 文件输出：`results_<策略名>.json`

---

### 3. 一键对比所有策略（推荐 ⭐）

```bash
python run_eval.py --compare-all
```

**这个命令会**：
1. 依次运行三种策略
2. 为每种策略生成独立的结果文件
3. **打印对比总结表格**，一目了然看到哪种策略最优

**示例输出**：
```
============================================================
COMPARISON SUMMARY
============================================================

Recall@5 by Strategy and Category:

Category                  fixed_200_40     semantic small_100_50 
--------------------------------------------------------------------------------
chunking_sensitive                0.80         0.90         0.85 
exact_match                       1.00         1.00         1.00 
semantic_paraphrase               1.00         1.00         1.00 
overall                           0.94         0.97         0.95 

✅ Comparison complete! Check results_*.json for details.
```

---

### 4. 常用参数

```bash
# 调整检索数量
python run_eval.py --chunking-strategy semantic --k 10

# 查看更多示例输出
python run_eval.py --chunking-strategy semantic --show-samples 10

# 批量对比时也可以调整参数
python run_eval.py --compare-all --k 10
```

---

## 预期结果

### Recall@5 对比（预估）

| 策略 | chunking_sensitive | exact_match | semantic_paraphrase | overall | 说明 |
|------|-------------------|-------------|---------------------|---------|------|
| **fixed_200_40** | 0.80 | 1.00 | 1.00 | 0.94 | Iteration 1 baseline |
| **semantic** | **0.90+** | 1.00 | 1.00 | **0.97+** | 预期：解决 ID 3（切块边界问题） |
| **small_100_50** | 0.85-0.90 | 1.00 | 1.00 | 0.95+ | 预期：大重叠减少边界问题 |

**关键观察点**：
1. **chunking_sensitive 类别**：这是本次迭代的核心关注点，预期 semantic 策略表现最好
2. **ID 3 案例**：Iteration 1 未命中，预期 semantic 策略能命中
3. **块数量**：small_100_50 的块数量应该约为 fixed_200_40 的 2 倍

---

## 关键实现细节

### 1. 多 Collection 支持

```python
# retrieval.py 中的改动
_collections = {}  # 从单一 collection 改为字典

def _get_chroma_collection(strategy: str = "fixed_200_40"):
    collection_name = f"rag_docs_{strategy}"  # 根据策略生成不同的 collection 名
    # ...
```

**为什么需要多 collection**：
- 不同策略的块数量、块大小不同，向量也不同
- 混用会导致检索错误（比如用 semantic 的查询去检索 fixed_200_40 的向量）

### 2. 语义边界切分的实现

```python
# chunking.py 中的 semantic_chunks()
sentence_pattern = r'[^。！？.!?]+[。！？.!?]+'  # 正则匹配完整句子
sentences = re.findall(sentence_pattern, text)

# 累积句子直到达到目标大小，但不超过最大值
if len(current_chunk) + len(sentence) > max_size:
    # 保存当前块，开始新块
```

**关键设计**：
- 优先保证语义完整性（不切断句子）
- 其次控制块大小（target_size=200, max_size=350）

### 3. 批量对比模式

```python
# run_eval.py 中的 --compare-all
if args.compare_all:
    for strategy in ["fixed_200_40", "semantic", "small_100_50"]:
        run_single_strategy(args, strategy)
    # 打印对比表格
```

---

## 已知情况 / 不是 bug

- **首次运行较慢**：每个策略第一次运行时需要生成向量并存储到各自的 collection
- **chroma_db/ 目录占用空间增大**：存储了三个 collection，约 3-15MB（取决于块数量）
- **semantic 策略的块大小不均匀**：这是设计特性，不是 bug（保持语义完整性）
- **--compare-all 输出较长**：因为要运行三次完整流程，可通过 `--show-samples 0` 减少示例输出

---

## 后续迭代接入点

- **Iteration 3（Hybrid Search）**：确定最优 chunking 策略后，在 `retrieval.py` 中添加 `retrieve_hybrid()` 函数，融合向量检索和 BM25 关键词检索
  
- **Iteration 4（Reranker）**：在检索后增加 rerank 步骤，对 top-20 重新排序后取 top-5

- **Iteration 5（LLM-as-Judge）**：自动化评估 faithfulness 和 answer relevance

---

## 常见问题

### Q1: 如何清空所有向量库重新索引？

```bash
rm -rf chroma_db/
python run_eval.py --compare-all
```

### Q2: 某个策略的结果异常差？

检查以下可能：
1. 查看终端输出的块数量统计（`Corpus: X chunks from Y docs`）
2. 确认是否有 "Indexing ... chunks" 的输出（说明正在重新索引）
3. 查看 `results_<策略名>.json` 中的详细结果

### Q3: 如何只重新运行某一个策略？

```bash
# 删除该策略的 collection
rm -rf chroma_db/  # 或者手动删除 chroma_db/ 下对应的子目录

# 重新运行
python run_eval.py --chunking-strategy semantic
```

### Q4: 三种策略的结果差异不大怎么办？

可能原因：
1. **测试集问题**：如果测试集中 chunking_sensitive 类别的案例不够，差异不明显
2. **文档特性**：你的文档可能本身就很适合固定长度切块
3. **Ground truth 标注**：标注范围可能需要调整

建议：
- 查看 `Iteration2.md` 中的失败案例分析
- 特别关注 ID 3 案例在三种策略下的表现

### Q5: 如何查看每种策略生成的块？

```python
# 在 Python 交互环境中
from chunking import build_corpus_chunks

chunks_fixed = build_corpus_chunks("corpus", strategy="fixed_200_40")
chunks_semantic = build_corpus_chunks("corpus", strategy="semantic")
chunks_small = build_corpus_chunks("corpus", strategy="small_100_50")

print(f"fixed_200_40: {len(chunks_fixed)} chunks")
print(f"semantic: {len(chunks_semantic)} chunks")
print(f"small_100_50: {len(chunks_small)} chunks")

# 查看某个文档的切块效果
doc1_chunks_semantic = [c for c in chunks_semantic if c['doc_id'] == 'doc1']
for c in doc1_chunks_semantic[:3]:  # 查看前3个块
    print(f"[{c['start']}-{c['end']}] {c['text'][:50]}...")
```

---

## 评估运行记录

运行以下命令记录 Iteration 2 的结果：

```bash
# 一键对比（推荐）
python run_eval.py --compare-all

# 或分别运行
python run_eval.py --chunking-strategy fixed_200_40
python run_eval.py --chunking-strategy semantic
python run_eval.py --chunking-strategy small_100_50
```

对比三个结果文件中的 `scores` 字段，填写 `Iteration2.md` 验收报告。

---

## 实验分析指南

完成实验后，建议按以下步骤分析结果：

1. **对比整体 Recall@K**：哪种策略的 overall 分数最高？
2. **关注 chunking_sensitive 类别**：是否达到了 0.90+ 的目标？
3. **分析 ID 3 案例**：semantic 策略是否解决了切块边界问题？
4. **块数量 vs 性能权衡**：small_100_50 的块数量增加了多少？性能提升是否值得这个代价？
5. **填写 Iteration2.md**：记录实验结果、失败案例分析、最优策略选择

---

## 下一步

确定最优 chunking 策略后，进入 **Iteration 3: Hybrid Search**，解决 Iteration 1 中的 ID 7 案例（术语不一致问题）。
