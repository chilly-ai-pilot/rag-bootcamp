"""
检索模块

Iteration 0 检索器：返回随机块，完全忽略查询内容。
Iteration 1: 实现真实的向量检索（bge-base-zh + ChromaDB）。
Iteration 2: 支持多种 chunking 策略，每种策略使用独立的 ChromaDB collection。
Iteration 3: 新增 BM25 关键词检索和 Hybrid Search（向量 + BM25 + RRF 融合）。

支持的策略：
- fixed_200_40: 200字符，40字符重叠
- semantic: 按句子边界切分
- fixed_100_50: 100字符，50字符重叠
"""
import random
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
import jieba
from rank_bm25 import BM25Okapi


def retrieve_random(query: str, chunks: List[Dict], k: int = 5, seed: int = None) -> List[Dict]:
    """随机检索策略（Iteration 0 基线）
    
    从所有块中随机选择 k 个，完全不考虑查询内容。
    用于建立性能下限基线。
    
    参数:
        query: 用户查询（在此函数中被忽略）
        chunks: 所有可检索的文档块列表
        k: 返回的块数量，默认 5
        seed: 随机数种子，用于结果可复现
    
    返回:
        随机选择的 k 个文档块列表
    """
    # 使用指定种子创建随机数生成器（保证可复现）
    rng = random.Random(seed)
    
    # 从所有块中随机抽样 k 个（如果总块数少于 k，则返回全部）
    return rng.sample(chunks, min(k, len(chunks)))


# --- 全局变量：延迟初始化 ---
_embedding_model = None
_chroma_client = None
_collections = {}  # 改为字典，存储多个 collection


def _get_embedding_model():
    """延迟加载 embedding 模型（单例模式）"""
    global _embedding_model
    if _embedding_model is None:
        print("Loading bge-base-zh model...")
        # 优先使用本地缓存，避免重复下载
        _embedding_model = SentenceTransformer('BAAI/bge-base-zh-v1.5', local_files_only=False)
    return _embedding_model


def _get_chroma_collection(strategy: str = "fixed_200_40"):
    """获取或创建指定策略的 ChromaDB collection（单例模式）
    
    每种 chunking 策略使用独立的 collection，避免混淆。
    
    参数:
        strategy: chunking 策略名称，用作 collection 名称的一部分
    
    返回:
        ChromaDB collection 对象
    """
    global _chroma_client, _collections
    
    if _chroma_client is None:
        # 初始化 ChromaDB 客户端（使用持久化存储）
        _chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # 根据策略生成 collection 名称
    collection_name = f"rag_docs_{strategy}"
    
    if collection_name not in _collections:
        # 获取或创建 collection
        _collections[collection_name] = _chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
    
    return _collections[collection_name]


def retrieve_vector(query: str, chunks: List[Dict], k: int = 5, strategy: str = "fixed_200_40") -> List[Dict]:
    """向量检索策略（Iteration 1/2 实现）
    
    使用 bge-base-zh 模型对块进行向量化，存储到 ChromaDB，
    然后根据查询的向量相似度返回 top-k 个最相关的块。
    
    Iteration 2 新增：支持指定 chunking 策略，不同策略使用不同的 collection。
    
    参数:
        query: 用户查询
        chunks: 所有可检索的文档块列表
        k: 返回的块数量，默认 5
        strategy: chunking 策略名称，默认 "fixed_200_40"
    
    返回:
        与查询最相关的 k 个文档块列表
    """
    # 加载 embedding 模型
    model = _get_embedding_model()
    
    # 获取对应策略的 ChromaDB collection
    collection = _get_chroma_collection(strategy)
    
    # 检查是否需要重新索引（collection 为空或 chunk 数量不匹配）
    current_count = collection.count()
    if current_count != len(chunks):
        print(f"Indexing {len(chunks)} chunks (strategy: {strategy}) into ChromaDB...")
        
        # 清空 collection（重新索引）
        if current_count > 0:
            # 获取所有 ID 并删除（新版 ChromaDB 不支持空 where）
            all_ids = collection.get()['ids']
            if all_ids:
                collection.delete(ids=all_ids)
        
        # 准备文档文本和元数据
        texts = [c["text"] for c in chunks]
        ids = [str(c["chunk_id"]) for c in chunks]
        metadatas = [
            {
                "doc_id": c["doc_id"],
                "start": c["start"],
                "end": c["end"],
                "chunk_id": c["chunk_id"]
            }
            for c in chunks
        ]
        
        # 生成 embeddings
        embeddings = model.encode(texts, show_progress_bar=True).tolist()
        
        # 批量插入到 ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"Indexed {len(chunks)} chunks for strategy '{strategy}'.")
    
    # 对查询进行向量化
    query_embedding = model.encode([query])[0].tolist()
    
    # 在 ChromaDB 中检索 top-k 相似文档
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, len(chunks))
    )
    
    # 重建返回的 chunk 格式（保持与 retrieve_random 相同的接口）
    retrieved_chunks = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        retrieved_chunks.append({
            "doc_id": metadata["doc_id"],
            "start": metadata["start"],
            "end": metadata["end"],
            "chunk_id": metadata["chunk_id"],
            "text": results["documents"][0][i]
        })
    
    return retrieved_chunks


# --- Iteration 3: BM25 关键词检索 ---

def retrieve_bm25(query: str, chunks: List[Dict], k: int = 5) -> List[Dict]:
    """BM25 关键词检索策略（Iteration 3 新增）
    
    使用 BM25 算法进行关键词匹配检索，特别适合处理：
    - 产品型号、编号等精确匹配场景
    - 专业术语、特定词汇
    - 需要关键词完全匹配的查询
    
    工作流程：
    1. 使用 jieba 对所有文档块进行中文分词
    2. 构建 BM25 索引
    3. 对查询进行分词
    4. 计算每个文档的 BM25 分数
    5. 返回得分最高的 top-k 文档
    
    参数:
        query: 用户查询
        chunks: 所有可检索的文档块列表
        k: 返回的块数量，默认 5
    
    返回:
        按 BM25 分数降序排列的 top-k 文档块列表
    """
    # 对所有文档块进行分词
    tokenized_corpus = [list(jieba.cut(chunk["text"])) for chunk in chunks]
    
    # 构建 BM25 索引
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 对查询进行分词
    tokenized_query = list(jieba.cut(query))
    
    # 计算所有文档的 BM25 分数
    scores = bm25.get_scores(tokenized_query)
    
    # 获取得分最高的 top-k 文档索引
    # argsort 返回从小到大的索引，所以用 [::-1] 反转为降序
    top_k_indices = scores.argsort()[::-1][:k]
    
    # 返回对应的文档块
    return [chunks[i] for i in top_k_indices]


# --- Iteration 3: Hybrid Search（向量 + BM25 + RRF 融合）---

def retrieve_hybrid(
    query: str, 
    chunks: List[Dict], 
    k: int = 5,
    strategy: str = "fixed_200_40",
    k_vector: int = 20,
    k_bm25: int = 20,
    rrf_k: int = 60
) -> List[Dict]:
    """混合检索策略（Iteration 3 新增）
    
    结合向量检索和 BM25 关键词检索的优势，使用 RRF（倒数排名融合）算法
    合并两种检索结果，兼顾语义相似性和关键词精确匹配。
    
    工作流程：
    1. 向量检索召回 top-20 文档（语义相关）
    2. BM25 检索召回 top-20 文档（关键词匹配）
    3. 使用 RRF 算法融合两个排序列表
    4. 返回融合后的 top-k 文档
    
    为什么这样设计：
    - 召回阶段扩大到 top-20：增加覆盖面，避免遗漏相关文档
    - RRF 融合：自动平衡两种检索的贡献，无需手动调参
    - 最终返回 top-k：满足下游生成器的输入要求
    
    参数:
        query: 用户查询
        chunks: 所有可检索的文档块列表
        k: 最终返回的块数量，默认 5
        strategy: chunking 策略名称（仅用于向量检索），默认 "fixed_200_40"
        k_vector: 向量检索召回数量，默认 20
        k_bm25: BM25 检索召回数量，默认 20
        rrf_k: RRF 算法的常数 k，默认 60（经验值）
    
    返回:
        融合后按 RRF 分数降序排列的 top-k 文档块列表
    """
    # 步骤 1: 向量检索召回 top-20
    vector_results = retrieve_vector(query, chunks, k=k_vector, strategy=strategy)
    
    # 步骤 2: BM25 检索召回 top-20
    bm25_results = retrieve_bm25(query, chunks, k=k_bm25)
    
    # 步骤 3: RRF 融合
    hybrid_results = _reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results,
        k=rrf_k,
        top_k=k
    )
    
    return hybrid_results


def _reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results: List[Dict],
    k: int = 60,
    top_k: int = 5
) -> List[Dict]:
    """倒数排名融合（Reciprocal Rank Fusion, RRF）算法
    
    RRF 是一种无需调参的融合算法，通过排名的倒数来合并多个检索器的结果。
    
    核心思想：
    - 不看具体分数，只看排名（避免量纲问题）
    - 每个文档的 RRF 分数 = Σ 1/(k + rank_i)
    - 排名越靠前，贡献越大
    
    优势：
    1. 无需调参：不需要设置向量和 BM25 的权重
    2. 量纲无关：向量相似度（0-1）和 BM25 分数（0-∞）可以直接融合
    3. 鲁棒性强：即使某个检索器表现差，也不会严重拖累整体
    4. 业界验证：Elasticsearch、OpenSearch 等生产系统都内置支持
    
    处理平局的策略：
    - 当两个文档的 RRF 分数相同时，优先保留在向量检索中排名更高的文档
    - 原因：语义理解（向量检索）通常比关键词匹配（BM25）更符合现代 RAG 的核心能力
    
    参数:
        vector_results: 向量检索结果（已按相似度降序排列）
        bm25_results: BM25 检索结果（已按分数降序排列）
        k: RRF 常数，默认 60（Gordon et al. 2006 论文的经验值）
        top_k: 最终返回的文档数量
    
    返回:
        融合后的 top-k 文档列表
    
    示例：
        假设两个文档在两个检索器中的排名：
        - doc_A: 向量排名1，BM25排名2
        - doc_B: 向量排名2，BM25排名1
        
        RRF 分数计算（k=60）：
        - doc_A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
        - doc_B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
        
        分数相同，按向量排名排序，doc_A 排在前面
    """
    # 记录每个文档在向量检索中的原始排名（用于打破平局）
    vector_ranks = {
        doc['chunk_id']: rank 
        for rank, doc in enumerate(vector_results, start=1)
    }
    
    # 步骤 1: 初始化 RRF 分数字典
    rrf_scores = {}
    
    # 步骤 2: 累加向量检索的贡献
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc['chunk_id']
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
    
    # 步骤 3: 累加 BM25 检索的贡献
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc['chunk_id']
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)
    
    # 步骤 4: 按 RRF 分数降序排序，分数相同时按向量排名升序排序
    sorted_docs = sorted(
        rrf_scores.items(),
        key=lambda x: (
            -x[1],  # RRF 分数降序（负号表示降序）
            vector_ranks.get(x[0], float('inf'))  # 分数相同时，向量排名升序（越小越好）
        )
    )
    
    # 步骤 5: 恢复完整文档对象
    # 创建 chunk_id 到文档对象的映射
    doc_map = {d['chunk_id']: d for d in vector_results + bm25_results}
    
    # 返回 top-k 文档（过滤掉可能的重复或缺失）
    result = [
        doc_map[doc_id] 
        for doc_id, score in sorted_docs[:top_k] 
        if doc_id in doc_map
    ]
    
    return result


# --- 实验性功能：混合粒度 Hybrid Search ---
# 如果效果不佳，可以整体删除此部分

def retrieve_hybrid_multi_granularity(
    query: str,
    corpus_dir: str,
    k: int = 5,
    strategy_vector: str = "fixed_100_50",
    strategy_bm25: str = "fixed_200_40",
    k_vector: int = 20,
    k_bm25: int = 20,
    rrf_k: int = 60,
    overlap_threshold: float = 0.5
) -> List[Dict]:
    """混合粒度 Hybrid Search（实验性）
    
    核心思想：
    - Vector 检索使用短块（fixed_100_50）→ 语义精确，性能 0.97
    - BM25 检索使用长块（fixed_200_40）→ 关键词丰富，性能 0.94
    - 理论上限：可能达到 > 0.97
    
    挑战：
    - 不同粒度的块需要对齐和去重
    - 重叠块的处理策略
    - 返回结果的统一
    
    参数:
        query: 用户查询
        corpus_dir: 语料库目录路径
        k: 最终返回的块数量
        strategy_vector: Vector 检索的 chunking 策略（默认 fixed_100_50）
        strategy_bm25: BM25 检索的 chunking 策略（默认 fixed_200_40）
        k_vector: Vector 召回数量
        k_bm25: BM25 召回数量
        rrf_k: RRF 常数
        overlap_threshold: 判断块重叠的阈值（默认 0.5 = 50%）
    
    返回:
        去重后的 top-k 文档块列表
    
    示例:
        results = retrieve_hybrid_multi_granularity(
            query="SmartCam-200 的声音报警音量怎么调节？",
            corpus_dir="corpus",
            k=5
        )
    """
    from chunking import build_corpus_chunks
    
    # 1. 构建两套不同粒度的块
    print(f"Building chunks: vector={strategy_vector}, bm25={strategy_bm25}")
    chunks_vector = build_corpus_chunks(corpus_dir, strategy=strategy_vector)
    chunks_bm25 = build_corpus_chunks(corpus_dir, strategy=strategy_bm25)
    
    print(f"  Vector chunks: {len(chunks_vector)}")
    print(f"  BM25 chunks: {len(chunks_bm25)}")
    
    # 2. 分别检索
    print(f"Retrieving with vector (top-{k_vector})...")
    vector_results = retrieve_vector(query, chunks_vector, k=k_vector, strategy=strategy_vector)
    
    print(f"Retrieving with BM25 (top-{k_bm25})...")
    bm25_results = retrieve_bm25(query, chunks_bm25, k=k_bm25)
    
    # 3. 标准化：转换为统一格式 (doc_id, start, end, text, source)
    def normalize_chunk(chunk, source):
        return {
            'doc_id': chunk['doc_id'],
            'start': chunk['start'],
            'end': chunk['end'],
            'text': chunk['text'],
            'source': source,  # 标记来源
            'length': chunk['end'] - chunk['start']
        }
    
    vector_normalized = [normalize_chunk(c, 'vector') for c in vector_results]
    bm25_normalized = [normalize_chunk(c, 'bm25') for c in bm25_results]
    
    # 4. 计算 RRF 分数（基于原始排名）
    rrf_scores = {}
    
    # Vector 贡献
    for rank, chunk in enumerate(vector_normalized, start=1):
        key = (chunk['doc_id'], chunk['start'], chunk['end'])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rrf_k + rank)
    
    # BM25 贡献
    for rank, chunk in enumerate(bm25_normalized, start=1):
        key = (chunk['doc_id'], chunk['start'], chunk['end'])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rrf_k + rank)
    
    # 5. 按 RRF 分数排序所有候选块
    all_candidates = vector_normalized + bm25_normalized
    
    # 去重：相同位置的块只保留一个
    unique_candidates = {}
    for chunk in all_candidates:
        key = (chunk['doc_id'], chunk['start'], chunk['end'])
        if key not in unique_candidates:
            unique_candidates[key] = chunk
            unique_candidates[key]['rrf_score'] = rrf_scores.get(key, 0)
    
    # 按 RRF 分数排序
    sorted_candidates = sorted(
        unique_candidates.values(),
        key=lambda x: x['rrf_score'],
        reverse=True
    )
    
    # 6. 去重：处理重叠块（核心逻辑）
    def calculate_overlap_ratio(chunk1, chunk2):
        """计算两个块的重叠比例"""
        if chunk1['doc_id'] != chunk2['doc_id']:
            return 0.0
        
        overlap_start = max(chunk1['start'], chunk2['start'])
        overlap_end = min(chunk1['end'], chunk2['end'])
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_len = overlap_end - overlap_start
        min_len = min(chunk1['length'], chunk2['length'])
        
        return overlap_len / min_len if min_len > 0 else 0.0
    
    # 贪心去重：按 RRF 分数从高到低，跳过与已选块重叠的块
    final_results = []
    
    for candidate in sorted_candidates:
        # 检查是否与已选块重叠
        is_overlapping = False
        
        for selected in final_results:
            overlap_ratio = calculate_overlap_ratio(candidate, selected)
            
            if overlap_ratio > overlap_threshold:
                is_overlapping = True
                
                # 如果候选块更长（包含更多上下文），替换已选块
                if candidate['length'] > selected['length']:
                    print(f"  Replace: {selected['source']} [{selected['start']}-{selected['end']}] RRF={selected['rrf_score']:.4f} "
                          f"with {candidate['source']} [{candidate['start']}-{candidate['end']}] RRF={candidate['rrf_score']:.4f} "
                          f"(overlap={overlap_ratio:.2f})")
                    final_results.remove(selected)
                    final_results.append(candidate)
                break
        
        if not is_overlapping:
            final_results.append(candidate)
        
        # 达到目标数量就停止
        if len(final_results) >= k:
            break
    
    # 7. 转换回标准格式
    result = []
    for chunk in final_results[:k]:
        result.append({
            'doc_id': chunk['doc_id'],
            'start': chunk['start'],
            'end': chunk['end'],
            'text': chunk['text'],
            'chunk_id': f"{chunk['doc_id']}_{chunk['start']}_{chunk['end']}",  # 生成唯一 ID
        })
    
    print(f"\nFinal results: {len(result)} chunks")
    print(f"  From vector: {sum(1 for c in final_results[:k] if c['source'] == 'vector')}")
    print(f"  From BM25: {sum(1 for c in final_results[:k] if c['source'] == 'bm25')}")
    
    return result


# --- Iteration 4: Reranker ---

from FlagEmbedding import FlagReranker

# 全局缓存 reranker 模型，避免重复加载
_reranker_model = None

def _get_reranker_model():
    """获取或初始化 reranker 模型（单例模式）
    
    使用 bge-reranker-base，这是一个 cross-encoder 架构的模型：
    - 同时编码 query 和 document，计算匹配分数
    - 比 bi-encoder（Vector 检索）精度更高，但速度更慢
    - 适合对召回结果进行精排
    
    返回:
        FlagReranker 模型实例
    """
    global _reranker_model
    if _reranker_model is None:
        print("Loading bge-reranker-base model...")
        from FlagEmbedding import FlagReranker
        
        _reranker_model = FlagReranker(
            'BAAI/bge-reranker-base',
            use_fp16=True  # 使用半精度加速，节省显存
        )
    return _reranker_model


def rerank_chunks(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """对已检索的文档块进行重新排序
    
    这是一个独立的 rerank 函数，可以应用于任何检索模式的结果。
    它使用 bge-reranker-base (cross-encoder) 对候选块重新打分。
    
    参数:
        query: 用户查询
        chunks: 已检索的文档块列表（来自任何检索模式）
        top_k: 返回的前 K 个结果数量
    
    返回:
        按 rerank 分数排序的 top-k 文档块列表，每个块包含 'rerank_score' 字段
    """
    if len(chunks) == 0:
        return []
    
    # 准备 reranker 输入
    reranker = _get_reranker_model()
    pairs = [[query, c['text']] for c in chunks]
    
    # 计算 rerank 分数（normalize=True: 归一化到 [0, 1]）
    scores = reranker.compute_score(pairs, normalize=True)
    
    # 为每个块添加 rerank 分数
    for i, chunk in enumerate(chunks):
        chunk['rerank_score'] = float(scores[i])
    
    # 按 rerank 分数降序排序
    reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
    
    # 返回 top-k
    return reranked[:top_k]


def retrieve_rerank(
    query: str,
    chunks: List[Dict],
    k: int = 5,
    strategy: str = "fixed_100_50",
    k_candidates: int = 20
) -> List[Dict]:
    """Rerank 检索策略（Iteration 4 核心实现）
    
    工作流程：
    1. 使用 Hybrid 检索召回 top-20 候选块（扩大召回范围）
    2. 使用 bge-reranker-base（cross-encoder）对候选块重新打分
    3. 按 rerank 分数排序，返回 top-5
    
    为什么需要 Reranker：
    - Hybrid 检索（RRF 融合）只看排名，不看内容
    - Reranker 使用 cross-encoder 理解 query-doc 的真实匹配度
    - 能过滤"看起来相关但实际不相关"的噪声块
    - 能识别产品型号、参数等精确匹配
    
    技术细节：
    - Cross-encoder vs Bi-encoder：
      * Bi-encoder（Vector 检索）：分别编码 query 和 doc，计算余弦相似度
      * Cross-encoder（Reranker）：同时编码 query + doc，直接预测相关性分数
      * Cross-encoder 更精确，但计算成本高（需要对每个 doc 单独计算）
    
    - 为什么先召回 20 再精排到 5：
      * 召回阶段用快速方法（Hybrid）扩大覆盖面
      * 精排阶段用精确方法（Reranker）保证质量
      * 平衡精度和效率
    
    参数:
        query: 用户查询
        chunks: 所有可检索的文档块列表
        k: 最终返回的块数量（默认 5）
        strategy: chunking 策略名称（传给 hybrid 检索），默认 "fixed_100_50"
        k_candidates: hybrid 召回的候选数量（默认 20）
    
    返回:
        按 rerank 分数排序的 top-k 文档块列表，每个块包含 'rerank_score' 字段
    
    示例:
        # 对比 hybrid 和 rerank
        hybrid_results = retrieve_hybrid(query, chunks, k=5)
        rerank_results = retrieve_rerank(query, chunks, k=5, k_candidates=20)
        
        # 查看 rerank 分数分布（为 Iteration 6 拒答阈值做准备）
        scores = [r['rerank_score'] for r in rerank_results]
        print(f"Rerank scores: min={min(scores):.4f}, max={max(scores):.4f}")
    """
    # Step 1: 使用 hybrid 检索召回候选块
    # 召回更多候选（20 个），给 reranker 更多选择空间
    candidates = retrieve_hybrid(
        query=query,
        chunks=chunks,
        k=k_candidates,
        strategy=strategy
    )
    
    if len(candidates) == 0:
        return []
    
    # Step 2: 准备 reranker 输入
    # FlagReranker 需要 [query, doc] 对的列表
    reranker = _get_reranker_model()
    pairs = [[query, c['text']] for c in candidates]
    
    # Step 3: 计算 rerank 分数
    # normalize=True: 将分数归一化到 [0, 1] 区间，方便设置阈值
    scores = reranker.compute_score(pairs, normalize=True)
    
    # Step 4: 为每个候选块添加 rerank 分数
    for i, candidate in enumerate(candidates):
        candidate['rerank_score'] = float(scores[i])
    
    # Step 5: 按 rerank 分数降序排序
    reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
    
    # Step 6: 返回 top-k
    return reranked[:k]
