"""
检索模块

Iteration 0 检索器：返回随机块，完全忽略查询内容。
Iteration 1: 实现真实的向量检索（bge-base-zh + ChromaDB）。
Iteration 2: 支持多种 chunking 策略，每种策略使用独立的 ChromaDB collection。

支持的策略：
- fixed_200_40: 200字符，40字符重叠
- semantic: 按句子边界切分
- small_100_50: 100字符，50字符重叠
"""
import random
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer


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
