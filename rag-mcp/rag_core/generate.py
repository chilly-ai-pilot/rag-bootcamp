"""
RAG Core - Generate Module

完整生成逻辑：检索 + 两段式生成 + 确定性校验。
用于 generate_answer MCP Tool。
支持完整的四层拒答机制（Layer 0-3）。
"""

import os
import sys
import asyncio
from typing import List, Dict, Optional

# 添加 iteration8 到路径
ITERATION8_PATH = os.path.join(os.path.dirname(__file__), '../../iteration8')
sys.path.insert(0, ITERATION8_PATH)

from chunking import build_corpus_chunks
from retrieval import retrieve_vector, retrieve_bm25, retrieve_hybrid, rerank_chunks
from generation import generate_answer_async
from scoring import hit, find_answer_rank


async def generate_answer_with_retrieval(
    query: str,
    top_k: int = 5,
    retrieval_mode: str = "hybrid",
    rerank: bool = True,
    corpus_dir: str = None,
    chunking_strategy: str = "fixed_100_50",
    rejection_config: Optional[Dict] = None
) -> Dict:
    """
    完整生成接口 - 检索 + 生成 + 校验
    
    Agent 用来获取最终答案。
    
    参数:
        query: 用户问题
        top_k: 检索片段数，默认 5
        retrieval_mode: 检索模式 (vector/bm25/hybrid)
        rerank: 是否使用 rerank
        corpus_dir: 语料库目录
        chunking_strategy: 分块策略
        rejection_config: 拒答配置（包含 judge_model 配置）
    
    返回:
        {
            "answer": str,              # 带引用标注的完整答案
            "citations": [              # 引用列表
                {
                    "span": str,        # 被引用的文字
                    "source": str,      # 来源标识
                    "chunk_id": str
                }
            ],
            "rejected": bool,           # 是否触发拒答
            "rejection_reason": str,    # 拒答原因（layer0-3）
            "faithfulness_score": float,# Judge 模型打分
            "relevance_score": float,   # Relevance 打分
            "query": str,
            "retrieved_count": int,     # 检索到的文档数
            "retrieval_rejected": bool  # 检索阶段是否被拒
        }
    """
    # 设置默认 corpus_dir
    if corpus_dir is None:
        corpus_dir = os.path.join(ITERATION8_PATH, 'corpus')
    
    # 构建文档块
    chunks = build_corpus_chunks(corpus_dir, strategy=chunking_strategy)
    
    # 检索阶段
    # 如果使用 rerank，先检索更多候选（top_k * 8），然后 rerank 到 top_k
    initial_top_k = top_k * 8 if rerank else top_k
    
    if retrieval_mode == "vector":
        retrieved = retrieve_vector(query, chunks, k=initial_top_k, strategy=chunking_strategy)
    elif retrieval_mode == "bm25":
        retrieved = retrieve_bm25(query, chunks, k=initial_top_k)
    elif retrieval_mode == "hybrid":
        retrieved = retrieve_hybrid(query, chunks, k=initial_top_k, strategy=chunking_strategy)
    else:
        raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")
    
    # Rerank（如果启用）
    if rerank:
        retrieved = rerank_chunks(query, retrieved, top_k=top_k)
    
    # 检查检索是否被拒（Layer 1）
    retrieval_rejected = False
    retrieval_rejection_reason = None
    
    if rejection_config and rejection_config.get('rejection_enabled', False):
        layer1_cfg = rejection_config.get('rejection_layers', {}).get('layer1_rerank', {})
        
        if layer1_cfg.get('enabled', False) and rerank and retrieved:
            # 检查 top-1 分数
            if 'rerank_score' in retrieved[0]:
                top1_score = retrieved[0]['rerank_score']
                top1_threshold = layer1_cfg.get('top1_threshold', 0.50)
                
                if top1_score < top1_threshold:
                    retrieval_rejected = True
                    retrieval_rejection_reason = f"[Layer 1] Top-1 rerank score too low (score={top1_score:.4f}, threshold={top1_threshold})"
            
            # 检查 top-3 平均分数
            if not retrieval_rejected and len(retrieved) >= 3:
                top3_scores = [c.get('rerank_score', 0) for c in retrieved[:3]]
                top3_avg = sum(top3_scores) / len(top3_scores)
                top3_threshold = layer1_cfg.get('top3_avg_threshold', 0.45)
                
                if top3_avg < top3_threshold:
                    retrieval_rejected = True
                    retrieval_rejection_reason = f"[Layer 1] Top-3 avg rerank score too low (avg={top3_avg:.4f}, threshold={top3_threshold})"
    
    # 如果检索被拒，直接返回拒答
    if retrieval_rejected:
        rejection_message = rejection_config.get('rejection_message', "抱歉，我在提供的资料中未找到足够充分的信息来准确回答您的问题。")
        return {
            "answer": rejection_message,
            "citations": [],
            "rejected": True,
            "rejection_reason": retrieval_rejection_reason,
            "faithfulness_score": None,
            "relevance_score": None,
            "query": query,
            "retrieved_count": len(retrieved),
            "retrieval_rejected": True
        }
    
    # 生成阶段（包含 Layer 2 和 Layer 3 的拒答检查）
    from openai import AsyncOpenAI
    
    # 创建客户端（用于生成）
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY")
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=60.0
    )
    
    # 调用生成函数（包含 Layer 2 Citation 校验和 Layer 3 Judge 评估）
    gen_result = await generate_answer_async(
        query,
        retrieved,
        chunking_strategy,
        client=client,
        rejection_config=rejection_config,
        corpus_dir=corpus_dir  # 传递 corpus_dir
    )
    
    await client.close()
    
    # 构建返回结果
    return {
        "answer": gen_result.get("answer", ""),
        "citations": gen_result.get("citations", []),
        "rejected": gen_result.get("rejected", False),
        "rejection_reason": gen_result.get("rejection_reason"),
        "faithfulness_score": gen_result.get("faithfulness_score"),
        "relevance_score": gen_result.get("relevance_score"),
        "query": query,
        "retrieved_count": len(retrieved),
        "retrieval_rejected": False
    }


def generate_answer(
    query: str,
    top_k: int = 5,
    retrieval_mode: str = "hybrid",
    rerank: bool = True,
    corpus_dir: str = None,
    chunking_strategy: str = "fixed_100_50",
    rejection_config: Optional[Dict] = None
) -> Dict:
    """
    同步包装器 - 用于非异步环境
    """
    return asyncio.run(
        generate_answer_with_retrieval(
            query=query,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            rerank=rerank,
            corpus_dir=corpus_dir,
            chunking_strategy=chunking_strategy,
            rejection_config=rejection_config
        )
    )


# 用于测试
if __name__ == "__main__":
    import json
    
    # 加载拒答配置
    config_path = os.path.join(ITERATION8_PATH, 'rejection_config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        rejection_config = json.load(f)
    
    # 添加 Judge 模型配置
    rejection_config['judge_model'] = 'deepseek-chat'
    rejection_config['judge_base_url'] = 'https://api.deepseek.com'
    rejection_config['judge_api_key'] = os.getenv('DEEPSEEK_API_KEY')
    
    # 测试生成
    result = generate_answer(
        query="SmartLock-100 如何生成临时密码？",
        top_k=5,
        retrieval_mode="hybrid",
        rerank=True,
        rejection_config=rejection_config
    )
    
    print(f"Query: {result['query']}")
    print(f"Retrieved: {result['retrieved_count']}")
    print(f"Rejected: {result['rejected']}")
    if result['rejection_reason']:
        print(f"Rejection reason: {result['rejection_reason']}")
    
    print(f"\nAnswer: {result['answer']}")
    print(f"\nCitations ({len(result['citations'])}):")
    for i, cit in enumerate(result['citations'], 1):
        print(f"  {i}. [{cit['span']}] from {cit['source']}")
    
    if result['faithfulness_score']:
        print(f"\nFaithfulness: {result['faithfulness_score']:.3f}")
    if result['relevance_score']:
        print(f"Relevance: {result['relevance_score']:.3f}")
