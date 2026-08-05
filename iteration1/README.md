# Iteration 1: Naive RAG Baseline

**目标**：实现最简单、最原始的 RAG 系统——使用真实的向量检索替换随机检索，作为后续所有优化的对照组。

**验收标准**：用 Iteration 0 的测试集跑一遍，记录检索命中率（Recall@K），与随机基线对比，证明向量检索确实有效。

---

## 核心改进

相比 Iteration 0，本次迭代的关键变化：

| 组件 | Iteration 0 | Iteration 1 | 说明 |
|------|-------------|-------------|------|
| 检索策略 | `retrieve_random` | `retrieve_vector` | 从随机检索升级到基于语义相似度的向量检索 |
| Embedding 模型 | 无 | `bge-base-zh-v1.5` | 使用 BAAI 开源的中文向量模型 |
| 向量数据库 | 无 | ChromaDB | 持久化向量存储，支持快速相似度检索 |
| 相似度度量 | 无 | 余弦相似度 | 标准的向量相似度计算方法 |

**保持不变**：
- Chunking 策略（固定长度 200 字，40 字重叠）
- 生成器（DeepSeek API）
- 评分指标（Recall@K）

---

## 技术选型说明

### 为什么选 bge-base-zh？

- **中文优化**：专门针对中文语义检索训练，比通用的多语言模型在中文场景表现更好
- **开源免费**：适合学习项目反复实验，无 API 调用成本
- **社区认可**：在 MTEB Chinese 排行榜上表现优异

### 为什么选 ChromaDB？

- **零运维**：本地启动，几行代码即可创建持久化向量库
- **学习友好**：相比 pgvector 需要先搭建 Postgres，ChromaDB 更适合快速验证想法
- **生产就绪**：后续可无缝迁移到生产级向量数据库（如 pgvector、Milvus）

---

## 文件结构

```
iteration1/
├── requirements.txt      新增：sentence-transformers, chromadb, torch
├── chunking.py           复用 iteration0（未改动）
├── retrieval.py          核心改动：实现 retrieve_vector()
├── generation.py         复用 iteration0（未改动）
├── scoring.py            复用 iteration0（未改动）
├── run_eval.py           改动：支持 --strategy 参数切换检索策略
├── corpus/               复用 iteration0 的测试集
├── chroma_db/            自动生成：ChromaDB 持久化存储目录
└── README.md             本文件
```

---

## 使用方法

### 1. 安装依赖

```bash
cd iteration1
pip install -r requirements.txt
```

**首次运行说明**：
- `sentence-transformers` 会自动下载 `bge-base-zh-v1.5` 模型（约 400MB）
- 下载位置：`~/.cache/huggingface/hub/`
- 如遇网络问题，可设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`

### 2. 运行向量检索（Iteration 1）

```bash
python run_eval.py --strategy vector
```

**首次运行**：会对所有文档块生成向量并存储到 ChromaDB（耗时约 10-30 秒）  
**后续运行**：直接从 ChromaDB 加载向量，几乎无额外耗时

### 3. 运行随机检索（Iteration 0 基线）

```bash
python run_eval.py --strategy random
```

用于对比验证向量检索的提升效果。

### 4. 常用参数

```bash
# 调整检索数量
python run_eval.py --strategy vector --k 10

# 调整 chunking 参数
python run_eval.py --strategy vector --chunk-size 150 --overlap 30

# 查看更多示例输出
python run_eval.py --strategy vector --show-samples 5

# 指定输出文件
python run_eval.py --strategy vector --out results_vector.json
```

---

## 预期结果

### Recall@5 对比（预估）

| 检索策略 | 整体 Recall@5 | 说明 |
|---------|--------------|------|
| random | ~15-20% | 理论上接近 k/N（5/总块数），实际略高因为有重叠 |
| vector | **60-80%** | 具体数值取决于测试集难度和文档质量 |

**如果向量检索 Recall@K 没有显著提升**（<40%），可能原因：
1. 测试集的 ground truth 标注有误
2. chunking 粒度不合适（块太大或太小）
3. 查询与文档的语义表达差异过大

---

## 关键实现细节

### 1. 延迟加载 + 单例模式

```python
_embedding_model = None  # 全局单例，避免重复加载模型

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
    return _embedding_model
```

**为什么这样设计**：
- Embedding 模型加载耗时（首次 ~5 秒），多次调用会浪费时间
- 模型占用内存（~500MB），重复加载会导致 OOM

### 2. 智能重索引

```python
current_count = collection.count()
if current_count != len(chunks):
    # 只在块数量变化时重新索引
    # 避免每次运行都重新生成向量
```

**触发重索引的情况**：
- 首次运行（collection 为空）
- 修改了 `--chunk-size` 或 `--overlap` 参数
- 更新了语料库文档

### 3. 接口兼容性

```python
# retrieve_random 和 retrieve_vector 保持相同的函数签名
def retrieve_vector(query: str, chunks: List[Dict], k: int = 5) -> List[Dict]:
    # 返回格式与 retrieve_random 完全一致
    # 便于在 run_eval.py 中无缝切换
```

---

## 已知情况 / 不是 bug

- **首次运行较慢**：需要下载模型 + 生成所有块的向量，属于正常现象
- **chroma_db/ 目录占用空间**：约 1-5MB（取决于块数量），这是向量持久化的代价
- **无需 API key**：向量检索在本地运行，不依赖任何在线服务
- **generation 仍可能输出 [MOCK]**：需要设置 `DEEPSEEK_API_KEY` 才能看到真实生成结果

---

## 后续迭代接入点

- **Iteration 2（Chunking 优化）**：在 `chunking.py` 中添加新的切块策略，保持返回格式不变，直接在 `run_eval.py` 中对比不同策略的 Recall@K
  
- **Iteration 3（Hybrid Search）**：在 `retrieval.py` 中添加 `retrieve_hybrid()` 函数，融合向量检索和 BM25 关键词检索

- **Iteration 4（Reranker）**：在检索后增加 rerank 步骤，对 top-20 重新排序后取 top-5

---

## 常见问题

### Q1: 如何清空向量库重新索引？

```bash
rm -rf chroma_db/
python run_eval.py --strategy vector
```

### Q2: 向量检索比随机检索还差？

检查以下可能：
1. 确认 `--strategy vector` 参数是否生效
2. 查看终端输出是否有 "Loading bge-base-zh model..." 和 "Indexing ... chunks"
3. 运行 `python -c "from sentence_transformers import SentenceTransformer; print('OK')"` 验证依赖安装

### Q3: 如何查看检索到的具体文档块？

修改 `run_eval.py` 中的 `--show-samples` 值，或直接查看 `results.json` 中的 `results` 数组。

### Q4: 为什么不直接用 LangChain/LlamaIndex？

这些框架把检索、chunking、prompt 拼接都封装了，初期使用会看不到每一步的具体行为。等对每一层都有手感后，再引入框架做工程化封装更合适。这是迭代计划中明确说明的设计决策。

---

## 评估运行记录

运行以下命令记录基线结果：

```bash
# 随机基线（Iteration 0）
python run_eval.py --strategy random --out results_random.json

# 向量检索（Iteration 1）
python run_eval.py --strategy vector --out results_vector.json
```

对比两个结果文件中的 `scores` 字段，验证向量检索的提升效果。
