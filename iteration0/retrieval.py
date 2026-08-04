"""
检索模块

Iteration 0 检索器：返回随机块，完全忽略查询内容。
这是有意为之——它是计划中提到的"什么都不做能得多少分"的基线。
这里的 Recall@K 应该接近 k/N，不会好。

Iteration 1 将用 retrieve_vector（bge-base-zh + ChromaDB）替换 retrieve_random。
保持相同的函数签名 (query, chunks, k) -> list[chunk]，
这样在切换时 run_eval.py 无需修改。
"""
import random
from typing import List, Dict


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


# --- Iteration 1 存根，尚未实现 ---
def retrieve_vector(query: str, chunks: List[Dict], k: int = 5) -> List[Dict]:
    """向量检索策略（Iteration 1 待实现）
    
    使用 bge-base-zh 模型对块进行向量化，存储到 ChromaDB，
    然后根据查询的向量相似度返回 top-k 个最相关的块。
    
    参数:
        query: 用户查询
        chunks: 所有可检索的文档块列表
        k: 返回的块数量，默认 5
    
    返回:
        与查询最相关的 k 个文档块列表
    
    抛出:
        NotImplementedError: 此函数尚未实现
    """
    raise NotImplementedError("Iteration 1: embed chunks with bge-base-zh, store in ChromaDB, query top-k")
