"""
RAG Core - Search Module

纯检索逻辑，不做生成。用于 search_knowledge MCP Tool。
支持多种检索模式和拒答机制（Layer 0/1）。
"""

import os
import sys
from typing import List, Dict, Optional

# 添加 iteration8 到路径以导入现有模块
ITERATION8_PATH = os.path.join(os.path.dirname(__file__), '../../iteration8')
sys.path.insert(0, ITERATION8_PATH)

from chunking import build_corpus_chunks
from retrieval import retrieve_vector, retrieve_bm25, retrieve_hybrid, rerank_chunks
from scoring import hit, find_answer_rank


def search_knowledge(
    query: str,
    top_k: int = 10,
    retrieval_mode: str = "hybrid",
    rerank: bool = True,
    rerank_top_k: int = None,
    corpus_dir: str = None,
    chunking_strategy: str = "fixed_100_50",
    rejection_config: Optional[Dict] = None
) -> Dict:
    """
    纯检索接口 - 只检索知识库，不做生成
    
    Agent 用来判断"知识库里有没有相关内容、够不够回答"。
    
    参数:
        query: 用户问题或搜索关键词
        top_k: 返回片段数，默认 10，最大 20
        retrieval_mode: 检索模式 (vector/bm25/hybrid)
        rerank: 是否使用 rerank
        rerank_top_k: rerank 后保留数量（如果 None，使用 top_k）
        corpus_dir: 语料库目录
        chunking_strategy: 分块策略
        rejection_config: 拒答配置
    
    返回:
        {
            "results": [
                {
                    "chunk_id": str,
                    "text": str,
                    "score": float,
                    "doc_id": str,
                    "metadata": {
                        "start": int,
                        "end": int,
                        "rerank_score": float (if rerank)
                    }
                }
            ],
            "total_retrieved": int,  # 粗筛候选集大小
            "returned": int,          # 实际返回数量
            "rejected": bool,         # 是否触发检索层拒答
            "rejection_reason": str,  # 拒答原因（layer0/layer1）
            "query": str
        }
    """
    # 参数校验
    if top_k > 20:
        top_k = 20
    
    if rerank_top_k is None:
        rerank_top_k = top_k
    
    # 设置默认 corpus_dir
    if corpus_dir is None:
        corpus_dir = os.path.join(ITERATION8_PATH, 'corpus')
    
    # 构建文档块
    chunks = build_corpus_chunks(corpus_dir, strategy=chunking_strategy)
    
    # 检索
    # 如果使用 rerank，先检索更多候选（top_k * 4），然后 rerank 到 top_k
    initial_top_k = top_k * 4 if rerank else top_k
    
    if retrieval_mode == "vector":
        retrieved = retrieve_vector(query, chunks, k=initial_top_k, strategy=chunking_strategy)
    elif retrieval_mode == "bm25":
        retrieved = retrieve_bm25(query, chunks, k=initial_top_k)
    elif retrieval_mode == "hybrid":
        retrieved = retrieve_hybrid(query, chunks, k=initial_top_k, strategy=chunking_strategy)
    else:
        raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")
    
    total_retrieved = len(retrieved)
    
    # Rerank（如果启用）
    if rerank:
        retrieved = rerank_chunks(query, retrieved, top_k=rerank_top_k)
    
    # Layer 0/1 拒答检查（基于 rerank 分数）
    rejected = False
    rejection_reason = None
    
    if rejection_config and rejection_config.get('rejection_enabled', False):
        layer1_cfg = rejection_config.get('rejection_layers', {}).get('layer1_rerank', {})
        
        if layer1_cfg.get('enabled', False) and rerank:
            # 检查 top-1 分数
            if retrieved and 'rerank_score' in retrieved[0]:
                top1_score = retrieved[0]['rerank_score']
                top1_threshold = layer1_cfg.get('top1_threshold', 0.50)
                
                if top1_score < top1_threshold:
                    rejected = True
                    rejection_reason = f"[Layer 1] Top-1 rerank score too low (score={top1_score:.4f}, threshold={top1_threshold})"
            
            # 检查 top-3 平均分数
            if not rejected and len(retrieved) >= 3:
                top3_scores = [c.get('rerank_score', 0) for c in retrieved[:3]]
                top3_avg = sum(top3_scores) / len(top3_scores)
                top3_threshold = layer1_cfg.get('top3_avg_threshold', 0.45)
                
                if top3_avg < top3_threshold:
                    rejected = True
                    rejection_reason = f"[Layer 1] Top-3 avg rerank score too low (avg={top3_avg:.4f}, threshold={top3_threshold})"
    
    # 构建返回结果
    results = []
    for i, chunk in enumerate(retrieved):
        result_item = {
            "chunk_id": f"{chunk['doc_id']}_chunk_{i}",
            "text": chunk['text'],
            "score": chunk.get('score', 0.0),  # 原始检索分数
            "doc_id": chunk['doc_id'],
            "metadata": {
                "start": chunk.get('start', 0),
                "end": chunk.get('end', 0),
            }
        }
        
        # 如果有 rerank 分数，也加上
        if 'rerank_score' in chunk:
            result_item['metadata']['rerank_score'] = chunk['rerank_score']
        
        results.append(result_item)
    
    return {
        "results": results,
        "total_retrieved": total_retrieved,
        "returned": len(results),
        "rejected": rejected,
        "rejection_reason": rejection_reason,
        "query": query
    }


# 用于测试
if __name__ == "__main__":
    # 测试搜索
    result = search_knowledge(
        query="SmartLock-100 如何生成临时密码？",
        top_k=5,
        retrieval_mode="hybrid",
        rerank=True
    )
    
    print(f"Query: {result['query']}")
    print(f"Total retrieved: {result['total_retrieved']}")
    print(f"Returned: {result['returned']}")
    print(f"Rejected: {result['rejected']}")
    if result['rejection_reason']:
        print(f"Rejection reason: {result['rejection_reason']}")
    
    print(f"\nTop 3 results:")
    for i, r in enumerate(result['results'][:3], 1):
        print(f"\n{i}. Doc: {r['doc_id']}, Score: {r['score']:.4f}")
        if 'rerank_score' in r['metadata']:
            print(f"   Rerank: {r['metadata']['rerank_score']:.4f}")
        print(f"   Text: {r['text'][:100]}...")
