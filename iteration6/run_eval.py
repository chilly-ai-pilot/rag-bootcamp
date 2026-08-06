"""
Iteration 5 评估运行脚本：集成 LLM-as-Judge 自动化评估

执行流程：查询 -> 检索 -> 生成 -> 评分 -> Faithfulness 评估

新增功能 (Iteration 5):
- 使用 Qwen 作为独立 Judge 评估 Faithfulness（忠实度）
- 评估 Answer Relevance（相关性）
- 分析 Faithfulness 与检索质量的相关性

使用方法:
    # 运行评估（会自动调用 Judge 评估）
    python run_eval.py --chunking-strategy fixed_200_40 --retrieval-mode rerank
    
    # 快速测试（只评估前几条）
    python run_eval.py --chunking-strategy fixed_200_40 --retrieval-mode rerank --max-eval 5
    
    # 不使用 Judge 评估（快速验证检索）
    python run_eval.py --chunking-strategy fixed_200_40 --retrieval-mode rerank --judge-mode none
    
    # 使用 Ragas Judge 评估
    python run_eval.py --chunking-strategy fixed_200_40 --retrieval-mode rerank --judge-mode ragas

注意：Judge 评估需要调用 LLM API，会增加运行时间和成本
"""
import os
import argparse
import json
import numpy as np
import asyncio
from typing import List, Dict

from chunking import build_corpus_chunks
from retrieval import retrieve_random, retrieve_vector, retrieve_bm25, retrieve_hybrid, rerank_chunks
from generation import generate_answer, generate_answer_v6, generate_answer_v6_async
from scoring import hit, find_answer_rank, aggregate_by_category, calculate_mrr, analyze_rerank_score_distribution
from evaluation import (
    llm_faithfulness_check, 
    llm_faithfulness_check_async,
    llm_relevance_check,
    llm_relevance_check_async,
    llm_combined_check,
    llm_combined_check_async,
    get_judge_llm, 
    ragas_faithfulness_check, 
    ragas_faithfulness_check_async
)


async def retrieve_and_generate_one(
    query_obj: Dict,
    chunks: List,
    args,
    chunking_strategy: str,
    retrieval_mode: str,
    deepseek_client=None
) -> Dict:
    """
    单个查询的 检索 + 生成 流程（异步）
    
    参数:
        query_obj: 查询对象
        chunks: 文档块列表
        args: 命令行参数
        chunking_strategy: chunking 策略
        retrieval_mode: 检索模式
        deepseek_client: DeepSeek 异步客户端
    
    返回:
        完整的结果对象（包含检索、生成、验证信息）
    """
    q = query_obj
    
    # 步骤 1: 检索
    if retrieval_mode == "random":
        retrieved = retrieve_random(q["query"], chunks, k=args.retrieval_top_k, seed=42)
    elif retrieval_mode == "vector":
        retrieved = retrieve_vector(q["query"], chunks, k=args.retrieval_top_k, strategy=chunking_strategy)
    elif retrieval_mode == "bm25":
        retrieved = retrieve_bm25(q["query"], chunks, k=args.retrieval_top_k)
    elif retrieval_mode == "hybrid":
        retrieved = retrieve_hybrid(q["query"], chunks, k=args.retrieval_top_k, strategy=chunking_strategy)
    else:
        raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")
    
    # 如果启用 rerank，对检索结果重新排序
    if args.rerank_mode == "bge":
        retrieved = rerank_chunks(q["query"], retrieved, top_k=args.rerank_top_k)
    
    # 步骤 2: 生成
    if args.generation_version == "v6":
        gen_result = await generate_answer_v6_async(
            q["query"],
            retrieved,
            client=deepseek_client,
            enable_validation=True,
            validation_threshold=args.validation_threshold
        )
        answer = gen_result["answer"]
    else:
        # V5 同步生成（在异步函数中同步调用）
        answer = generate_answer(q["query"], retrieved)
        gen_result = None
    
    # 步骤 3: 评估检索质量
    h = hit(retrieved, q["doc_id"], q["char_start"], q["char_end"])
    answer_rank = find_answer_rank(retrieved, q["doc_id"], q["char_start"], q["char_end"])
    
    # 构建结果
    result_item = {
        "id": q["id"],
        "query": q["query"],
        "category": q["category"],
        "hit": h,
        "answer_rank": answer_rank,
        "answer": answer,
        "retrieved": retrieved,
    }
    
    # 添加 V6 特有字段
    if gen_result:
        result_item.update({
            "raw_answer": gen_result["raw_answer"],
            "citations": gen_result["citations"],
            "validation": gen_result["validation"],
            "llm_raw_response": gen_result["llm_raw_response"]
        })
    
    # 添加 rerank 分数
    if args.rerank_mode == "bge" and retrieved:
        result_item["rerank_scores"] = [c.get('rerank_score', 0.0) for c in retrieved]
    
    return result_item


async def batch_retrieve_and_generate(
    queries: List[Dict],
    chunks: List,
    args,
    chunking_strategy: str,
    retrieval_mode: str
) -> List[Dict]:
    """
    批量异步执行 检索+生成（每批 N 个并发）
    
    参数:
        queries: 查询列表
        chunks: 文档块列表
        args: 命令行参数
        chunking_strategy: chunking 策略
        retrieval_mode: 检索模式
    
    返回:
        结果列表
    """
    total = len(queries)
    print(f"🚀 Batch retrieve+generate for {total} queries (concurrency {args.batch_size})...")
    
    # 初始化 DeepSeek 客户端（如果需要）
    deepseek_client = None
    if args.generation_version == "v6":
        from openai import AsyncOpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key:
            deepseek_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
        else:
            print("⚠️  DEEPSEEK_API_KEY not set, will use mock generation")
    
    all_results = []
    
    # 分批并发执行
    batch_size = args.batch_size
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_queries = queries[batch_start:batch_end]
        
        print(f"   Processing batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch_queries)} queries)...")
        
        # 创建当前批次的任务（检索+生成）
        tasks = [
            retrieve_and_generate_one(q, chunks, args, chunking_strategy, retrieval_mode, deepseek_client)
            for q in batch_queries
        ]
        
        # 并发执行当前批次
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                print(f"⚠️  Query {batch_queries[i]['id']} failed: {result}")
                # 创建错误结果
                result = {
                    "id": batch_queries[i]["id"],
                    "query": batch_queries[i]["query"],
                    "category": batch_queries[i]["category"],
                    "hit": False,
                    "answer_rank": None,
                    "answer": f"[ERROR] {result}",
                    "retrieved": [],
                }
            all_results.append(result)
    
    # 关闭客户端
    if deepseek_client:
        await deepseek_client.close()
    
    print(f"✅ Batch retrieve+generate completed for {total} queries")
    
    return all_results


async def batch_evaluate_combined(results: List[Dict], args) -> List[Dict]:
    """
    批量异步评估 Faithfulness 和 Answer Relevance（组合评估，节省成本）
    
    一次 API 调用同时返回两个指标，节省 50% 的成本
    
    参数:
        results: 包含 query, answer 和 retrieved 的结果列表
        args: 命令行参数
    
    返回:
        更新后的 results（添加 faithfulness_score, relevance_score 等）
    """
    # 收集需要评估的查询
    eval_items = []
    for i, result in enumerate(results):
        if 'answer' in result and result['answer'] and 'retrieved' in result and result['retrieved']:
            eval_items.append((i, result))
    
    total = len(eval_items)
    judge_mode_names = {"llm": "LLM Judge", "ragas": "Ragas Judge"}
    judge_type = judge_mode_names.get(args.judge_mode, "Unknown")
    print(f"🚀 Combined evaluation for {total} queries with {judge_type} (batch size {args.batch_size})...")
    print(f"   Evaluating Faithfulness + Relevance in single API calls (50% cost savings)")
    
    from openai import AsyncOpenAI
    
    api_key = os.getenv("ALI_API_KEY")
    base_url = os.getenv("ALI_BASE_URL")
    
    if not api_key or not base_url:
        raise ValueError("请设置环境变量 ALI_API_KEY 和 ALI_BASE_URL")
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # 分批处理
    batch_size = args.batch_size
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_items = eval_items[batch_start:batch_end]
        
        print(f"   Processing batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch_items)} queries)...")
        
        # 创建当前批次的任务
        tasks = []
        indices = []
        for idx, result in batch_items:
            tasks.append(
                llm_combined_check_async(
                    result['query'],
                    result['answer'],
                    result['retrieved'],
                    client
                )
            )
            indices.append(idx)
        
        # 并发执行当前批次
        combined_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for idx, combined_result in zip(indices, combined_results):
            if isinstance(combined_result, Exception):
                print(f"⚠️  Query {results[idx]['id']} combined evaluation failed: {combined_result}")
                results[idx]["faithfulness_score"] = None
                results[idx]["relevance_score"] = None
                results[idx]["judge_response"] = f"Error: {combined_result}"
                continue
            
            # 提取两个分数
            response_text = combined_result['raw_response']
            import re
            
            # 提取 Faithfulness 分数
            faith_patterns = [
                r'【Faithfulness 分数】\s*\n?\s*([0-9.]+)',
                r'Faithfulness.*?分数.*?[:：]\s*([0-9.]+)',
            ]
            
            faithfulness_score = None
            for pattern in faith_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    try:
                        score = float(match.group(1))
                        if score > 1:
                            score = score / 100
                        faithfulness_score = score
                        break
                    except:
                        continue
            
            # 提取 Relevance 分数
            rel_patterns = [
                r'【Relevance 分数】\s*\n?\s*([0-9.]+)',
                r'Relevance.*?分数.*?[:：]\s*([0-9.]+)',
                r'【Answer Relevance 分数】\s*\n?\s*([0-9.]+)',
            ]
            
            relevance_score = None
            for pattern in rel_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    try:
                        score = float(match.group(1))
                        if score > 1:
                            score = score / 100
                        relevance_score = score
                        break
                    except:
                        continue
            
            # 如果没找到，尝试通用模式
            if faithfulness_score is None or relevance_score is None:
                all_scores = re.findall(r'\b([0-9]\.[0-9]{1,2})\b', response_text)
                if len(all_scores) >= 2:
                    if faithfulness_score is None:
                        faithfulness_score = float(all_scores[0])
                    if relevance_score is None:
                        relevance_score = float(all_scores[1])
            
            # 保存结果
            results[idx]["faithfulness_score"] = faithfulness_score if faithfulness_score is not None else 0.5
            results[idx]["relevance_score"] = relevance_score if relevance_score is not None else 0.5
            results[idx]["judge_response"] = response_text
    
    await client.close()
    
    return results


async def batch_evaluate_faithfulness(results: List[Dict], args) -> List[Dict]:
    """
    批量异步评估 Faithfulness，显著加速评估过程
    
    注意：推荐使用 batch_evaluate_combined 来节省 API 成本
    
    参数:
        results: 包含 answer 和 retrieved 的结果列表
        args: 命令行参数（包含 judge_mode 标志）
    
    返回:
        更新后的 results（添加 faithfulness_score 和 judge_response）
    """
    # 收集需要评估的查询
    eval_items = []
    for i, result in enumerate(results):
        if 'retrieved' in result and result['retrieved']:
            eval_items.append((i, result))
    
    total = len(eval_items)
    judge_mode_names = {"llm": "LLM Judge", "ragas": "Ragas Judge"}
    judge_type = judge_mode_names.get(args.judge_mode, "Unknown")
    print(f"🚀 Evaluating {total} queries with {judge_type} (batch size {args.batch_size})...")
    
    if args.judge_mode == "ragas":
        # 使用 Ragas Judge（异步）
        batch_size = args.batch_size
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_items = eval_items[batch_start:batch_end]
            
            print(f"   Processing batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch_items)} queries)...")
            
            # 创建当前批次的任务
            tasks = []
            indices = []
            for idx, result in batch_items:
                tasks.append(
                    ragas_faithfulness_check_async(
                        result['query'],
                        result['answer'],
                        result['retrieved']
                    )
                )
                indices.append(idx)
            
            # 并发执行当前批次
            judge_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for idx, judge_result in zip(indices, judge_results):
                if isinstance(judge_result, Exception):
                    print(f"⚠️  Query {results[idx]['id']} failed: {judge_result}")
                    results[idx]["faithfulness_score"] = None
                    results[idx]["judge_response"] = f"Error: {judge_result}"
                    continue
                
                results[idx]["faithfulness_score"] = judge_result.get('faithfulness_score')
                results[idx]["judge_response"] = judge_result.get('ragas_response', '')
    
    else:
        # 使用 LLM Judge（异步，args.judge_mode == "llm"）
        from openai import AsyncOpenAI
        
        api_key = os.getenv("ALI_API_KEY")
        base_url = os.getenv("ALI_BASE_URL")
        
        if not api_key or not base_url:
            raise ValueError("请设置环境变量 ALI_API_KEY 和 ALI_BASE_URL")
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        # 分批处理（避免并发过大触发限流）
        batch_size = args.batch_size
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_items = eval_items[batch_start:batch_end]
            
            print(f"   Processing batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch_items)} queries)...")
            
            # 创建当前批次的任务
            tasks = []
            indices = []
            for idx, result in batch_items:
                tasks.append(
                    llm_faithfulness_check_async(
                        result['answer'],
                        result['retrieved'],
                        client
                    )
                )
                indices.append(idx)
            
            # 并发执行当前批次
            judge_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for idx, judge_result in zip(indices, judge_results):
                if isinstance(judge_result, Exception):
                    print(f"⚠️  Query {results[idx]['id']} failed: {judge_result}")
                    results[idx]["faithfulness_score"] = None
                    continue
                
                # 提取分数
                response_text = judge_result['raw_response']
                import re
                
                score_patterns = [
                    r'【Faithfulness 分数】\s*\n?\s*([0-9.]+)',
                    r'Faithfulness.*?分数.*?[:：]\s*([0-9.]+)',
                    r'最终分数[:：]\s*([0-9.]+)',
                    r'总分[:：]\s*([0-9.]+)',
                ]
                
                faithfulness_score = None
                for pattern in score_patterns:
                    match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        try:
                            score = float(match.group(1))
                            if score > 1:
                                score = score / 100
                            faithfulness_score = score
                            break
                        except:
                            continue
                
                if faithfulness_score is None:
                    all_scores = re.findall(r'\b([0-9]\.[0-9]{1,2})\b', response_text)
                    if all_scores:
                        faithfulness_score = float(all_scores[-1])
                    else:
                        faithfulness_score = 0.5
                
                results[idx]["faithfulness_score"] = faithfulness_score
                results[idx]["judge_response"] = judge_result['raw_response']
        
        await client.close()
    
    return results


async def batch_evaluate_relevance(results: List[Dict], args) -> List[Dict]:
    """
    批量异步评估 Answer Relevance
    
    参数:
        results: 包含 query 和 answer 的结果列表
        args: 命令行参数（包含 judge_mode 标志和 batch_size）
    
    返回:
        更新后的 results（添加 relevance_score 和 relevance_response）
    """
    # 收集需要评估的查询
    eval_items = []
    for i, result in enumerate(results):
        if 'answer' in result and result['answer']:
            eval_items.append((i, result))
    
    total = len(eval_items)
    print(f"🚀 Evaluating Answer Relevance for {total} queries (batch size {args.batch_size})...")
    
    from openai import AsyncOpenAI
    
    api_key = os.getenv("ALI_API_KEY")
    base_url = os.getenv("ALI_BASE_URL")
    
    if not api_key or not base_url:
        raise ValueError("请设置环境变量 ALI_API_KEY 和 ALI_BASE_URL")
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # 分批处理
    batch_size = args.batch_size
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_items = eval_items[batch_start:batch_end]
        
        print(f"   Processing batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch_items)} queries)...")
        
        # 创建当前批次的任务
        tasks = []
        indices = []
        for idx, result in batch_items:
            tasks.append(
                llm_relevance_check_async(
                    result['query'],
                    result['answer'],
                    client
                )
            )
            indices.append(idx)
        
        # 并发执行当前批次
        relevance_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for idx, rel_result in zip(indices, relevance_results):
            if isinstance(rel_result, Exception):
                print(f"⚠️  Query {results[idx]['id']} relevance check failed: {rel_result}")
                results[idx]["relevance_score"] = None
                results[idx]["relevance_response"] = f"Error: {rel_result}"
                continue
            
            # 提取分数
            response_text = rel_result['raw_response']
            import re
            
            score_patterns = [
                r'【Answer Relevance 分数】\s*\n?\s*([0-9.]+)',
                r'Answer Relevance.*?分数.*?[:：]\s*([0-9.]+)',
                r'相关性分数[:：]\s*([0-9.]+)',
                r'Relevance Score[:：]\s*([0-9.]+)',
            ]
            
            relevance_score = None
            for pattern in score_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    try:
                        score = float(match.group(1))
                        if score > 1:
                            score = score / 100
                        relevance_score = score
                        break
                    except:
                        continue
            
            if relevance_score is None:
                # 尝试提取任何 0.xx 格式的分数
                all_scores = re.findall(r'\b([0-9]\.[0-9]{1,2})\b', response_text)
                if all_scores:
                    relevance_score = float(all_scores[-1])
                else:
                    relevance_score = 0.5
            
            results[idx]["relevance_score"] = relevance_score
            results[idx]["relevance_response"] = response_text
    
    await client.close()
    
    return results


def run_single_strategy(args, chunking_strategy, retrieval_mode=None):
    """运行单个 chunking 策略和检索模式的评估
    
    参数:
        args: 命令行参数
        chunking_strategy: chunking 策略名称
        retrieval_mode: 检索模式（如果为 None，使用 args.retrieval_mode）
    
    返回:
        (scores, results): 分类评分和详细结果
    """
    if retrieval_mode is None:
        retrieval_mode = args.retrieval_mode
    
    print(f"\n{'='*60}")
    print(f"Running evaluation:")
    print(f"  Chunking: {chunking_strategy}")
    print(f"  Retrieval: {retrieval_mode}")
    print(f"  Rerank: {args.rerank_mode}")
    print(f"  Judge Mode: {args.judge_mode}")
    print(f"{'='*60}")
    
    # 初始化 Judge LLM（如果需要）
    judge_llm = None
    if args.judge_mode != "none":
        try:
            print("\n📡 Initializing Judge LLM...")
            judge_llm = get_judge_llm()
            print("✅ Judge LLM ready")
        except Exception as e:
            print(f"⚠️  Judge LLM initialization failed: {e}")
            print("   Continuing without Judge evaluation...")
            args.judge_mode = "none"
    
    # 加载查询集
    with open(args.query_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    # 构建文档块索引（使用指定的 chunking 策略）
    chunks = build_corpus_chunks(args.corpus_dir, strategy=chunking_strategy)
    
    # 统计块信息
    num_docs = len(set(c['doc_id'] for c in chunks))
    avg_chunk_size = sum(len(c['text']) for c in chunks) / len(chunks)
    print(f"Corpus: {len(chunks)} chunks from {num_docs} docs")
    print(f"Average chunk size: {avg_chunk_size:.1f} characters")

    # ========================================
    # 批量异步执行：检索 + 生成（每批 N 个并发）
    # ========================================
    print(f"\n{'='*60}")
    print("Batch Retrieve + Generate (async)")
    print(f"{'='*60}")
    
    results = asyncio.run(
        batch_retrieve_and_generate(
            queries,
            chunks,
            args,
            chunking_strategy,
            retrieval_mode
        )
    )
    
    # 打印一些示例
    print(f"\n{'='*60}")
    print("Sample Results")
    print(f"{'='*60}")
    for result in results[:3]:
        print(f"\n--- Query {result['id']} [{result['category']}] ---")
        print(f"Q: {result['query']}")
        print(f"Hit: {result['hit']}")
        print(f"A: {result['answer'][:200]}...")
        if args.generation_version == "v6" and result.get('validation'):
            stats = result['validation'].get('stats', {})
            print(f"Citations: {stats.get('passed', 0)}/{stats.get('total', 0)} passed")
    
    # Iteration 5: 批量异步评估 Faithfulness（如果启用 Judge）
    if args.judge_mode != "none" and judge_llm:
        print(f"\n{'='*60}")
        print("Starting batch Judge evaluation (async)...")
        print(f"{'='*60}\n")
        
        try:
            # 使用组合评估（一次 API 调用返回两个指标，节省 50% 成本）
            results = asyncio.run(batch_evaluate_combined(results, args))
            
            # 打印 Faithfulness 分数示例
            judge_name = "Ragas Judge" if args.judge_mode == "ragas" else "LLM Judge"
            print(f"\n{'='*60}")
            print(f"Faithfulness Scores ({judge_name}):")
            print(f"{'='*60}")
            for result in results:
                score = result.get('faithfulness_score')
                if score is not None:
                    print(f"Query {result['id']}: {score:.2f}")
                    if score < 0.5:
                        print(f"  ⚠️  Low score warning!")
            
            # 打印 Relevance 分数示例
            print(f"\n{'='*60}")
            print(f"Answer Relevance Scores:")
            print(f"{'='*60}")
            for result in results:
                score = result.get('relevance_score')
                if score is not None:
                    print(f"Query {result['id']}: {score:.2f}")
                    if score < 0.5:
                        print(f"  ⚠️  Low relevance warning!")
                        
        except Exception as e:
            print(f"⚠️  Batch Judge evaluation failed: {e}")
            print("   Continuing without Judge evaluation...")
            args.judge_mode = "none"

    # 计算并打印分类别的 Recall@K 和 MRR 指标
    scores = aggregate_by_category(results)
    mrr_scores = calculate_mrr(results)
    
    print(f"\n=== Recall@{args.retrieval_top_k} (chunking: {chunking_strategy}, retrieval: {retrieval_mode}) ===")
    for cat, score in scores.items():
        print(f"  {cat:24s} {score:.2f}")
    
    print(f"\n=== MRR (Mean Reciprocal Rank) ===")
    for cat, score in mrr_scores.items():
        print(f"  {cat:24s} {score:.4f}")
    
    # Iteration 5: 分析 Faithfulness 分数
    faithfulness_analysis = None
    
    if args.judge_mode != "none":
        faithfulness_scores = [r.get('faithfulness_score') for r in results if r.get('faithfulness_score') is not None]
        
        if faithfulness_scores:
            faithfulness_analysis = {
                "count": len(faithfulness_scores),
                "mean": float(np.mean(faithfulness_scores)),
                "median": float(np.median(faithfulness_scores)),
                "min": float(np.min(faithfulness_scores)),
                "max": float(np.max(faithfulness_scores)),
                "std": float(np.std(faithfulness_scores))
            }
            
            # 按命中/未命中分组分析
            hit_faithfulness = [r.get('faithfulness_score') for r in results if r.get('hit') and r.get('faithfulness_score') is not None]
            miss_faithfulness = [r.get('faithfulness_score') for r in results if not r.get('hit') and r.get('faithfulness_score') is not None]
            
            if hit_faithfulness:
                faithfulness_analysis["hit_mean"] = float(np.mean(hit_faithfulness))
                faithfulness_analysis["hit_count"] = len(hit_faithfulness)
            
            if miss_faithfulness:
                faithfulness_analysis["miss_mean"] = float(np.mean(miss_faithfulness))
                faithfulness_analysis["miss_count"] = len(miss_faithfulness)
            
            judge_name = {"llm": "LLM Judge", "ragas": "Ragas Judge"}[args.judge_mode]
            print(f"\n=== Faithfulness Scores ({judge_name}) ===")
            print(f"  Evaluated queries:       {faithfulness_analysis['count']}")
            print(f"  Mean:                    {faithfulness_analysis['mean']:.3f}")
            print(f"  Median:                  {faithfulness_analysis['median']:.3f}")
            print(f"  Range:                   [{faithfulness_analysis['min']:.3f}, {faithfulness_analysis['max']:.3f}]")
            print(f"  Std Dev:                 {faithfulness_analysis['std']:.3f}")
            
            if "hit_mean" in faithfulness_analysis and "miss_mean" in faithfulness_analysis:
                print(f"\n  Hit queries ({faithfulness_analysis['hit_count']}):     {faithfulness_analysis['hit_mean']:.3f}")
                print(f"  Miss queries ({faithfulness_analysis['miss_count']}):    {faithfulness_analysis['miss_mean']:.3f}")
                print(f"  Delta:                   {faithfulness_analysis['hit_mean'] - faithfulness_analysis['miss_mean']:.3f}")
    
    # Iteration 5: 分析 Answer Relevance 分数
    relevance_analysis = None
    
    if args.judge_mode != "none":
        relevance_scores = [r.get('relevance_score') for r in results if r.get('relevance_score') is not None]
        
        if relevance_scores:
            relevance_analysis = {
                "count": len(relevance_scores),
                "mean": float(np.mean(relevance_scores)),
                "median": float(np.median(relevance_scores)),
                "min": float(np.min(relevance_scores)),
                "max": float(np.max(relevance_scores)),
                "std": float(np.std(relevance_scores))
            }
            
            # 按命中/未命中分组分析
            hit_relevance = [r.get('relevance_score') for r in results if r.get('hit') and r.get('relevance_score') is not None]
            miss_relevance = [r.get('relevance_score') for r in results if not r.get('hit') and r.get('relevance_score') is not None]
            
            if hit_relevance:
                relevance_analysis["hit_mean"] = float(np.mean(hit_relevance))
                relevance_analysis["hit_count"] = len(hit_relevance)
            
            if miss_relevance:
                relevance_analysis["miss_mean"] = float(np.mean(miss_relevance))
                relevance_analysis["miss_count"] = len(miss_relevance)
            
            print(f"\n=== Answer Relevance Scores ===")
            print(f"  Evaluated queries:       {relevance_analysis['count']}")
            print(f"  Mean:                    {relevance_analysis['mean']:.3f}")
            print(f"  Median:                  {relevance_analysis['median']:.3f}")
            print(f"  Range:                   [{relevance_analysis['min']:.3f}, {relevance_analysis['max']:.3f}]")
            print(f"  Std Dev:                 {relevance_analysis['std']:.3f}")
            
            if "hit_mean" in relevance_analysis and "miss_mean" in relevance_analysis:
                print(f"\n  Hit queries ({relevance_analysis['hit_count']}):     {relevance_analysis['hit_mean']:.3f}")
                print(f"  Miss queries ({relevance_analysis['miss_count']}):    {relevance_analysis['miss_mean']:.3f}")
                print(f"  Delta:                   {relevance_analysis['hit_mean'] - relevance_analysis['miss_mean']:.3f}")
    
    # Iteration 6: 分析引用验证统计
    validation_analysis = None
    
    if args.generation_version == "v6":
        # 收集所有 validation stats
        validation_stats_list = [
            r.get('validation', {}).get('stats', {})
            for r in results
            if r.get('validation', {}).get('enabled', False)
        ]
        
        if validation_stats_list:
            total_citations = sum(s.get('total', 0) for s in validation_stats_list)
            passed_citations = sum(s.get('passed', 0) for s in validation_stats_list)
            failed_citations = sum(s.get('failed', 0) for s in validation_stats_list)
            
            # 收集所有失败的 citations
            all_failed = []
            for r in results:
                failed_list = r.get('validation', {}).get('failed', [])
                for fail in failed_list:
                    all_failed.append({
                        "query_id": r['id'],
                        "query": r['query'],
                        **fail
                    })
            
            validation_analysis = {
                "total_queries": len(validation_stats_list),
                "total_citations": total_citations,
                "passed_citations": passed_citations,
                "failed_citations": failed_citations,
                "pass_rate": passed_citations / total_citations if total_citations > 0 else 0.0,
                "failed_details": all_failed
            }
            
            print(f"\n=== Citation Validation Statistics (Iteration 6) ===")
            print(f"  Total queries evaluated: {validation_analysis['total_queries']}")
            print(f"  Total citations:         {validation_analysis['total_citations']}")
            print(f"  Passed:                  {validation_analysis['passed_citations']} ({validation_analysis['pass_rate']:.1%})")
            print(f"  Failed:                  {validation_analysis['failed_citations']} ({1 - validation_analysis['pass_rate']:.1%})")
            
            if all_failed:
                print(f"\n  Failed citations by reason:")
                reason_counts = {}
                for fail in all_failed:
                    reason = fail['reason']
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                
                for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    • {reason}: {count}")
                
                # 显示前几个失败案例
                print(f"\n  Sample failed citations (first 3):")
                for fail in all_failed[:3]:
                    print(f"    Query {fail['query_id']}: '{fail['span']}' → {fail['source']}")
                    print(f"      Reason: {fail['reason']}")
    
    # 清理 results 中的 retrieved 字段（太大不需要保存）
    for r in results:
        r.pop('retrieved', None)
    
    # 如果使用 rerank 模式，分析 rerank 分数分布
    rerank_analysis = None
    if retrieval_mode == "rerank":
        rerank_analysis = analyze_rerank_score_distribution(results)
        if "error" not in rerank_analysis:
            print(f"\n=== Rerank Score Distribution (for Iteration 6) ===")
            stats = rerank_analysis["statistics"]
            print(f"  Total scores:            {stats['total_scores']}")
            print(f"  Range:                   [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Mean:                    {stats['mean']:.4f}")
            print(f"  Median:                  {stats['median']:.4f}")
            print(f"  P25/P75:                 [{stats['p25']:.4f}, {stats['p75']:.4f}]")
            print(f"  P90/P95:                 [{stats['p90']:.4f}, {stats['p95']:.4f}]")
            
            if "hit_statistics" in rerank_analysis and rerank_analysis["hit_statistics"]:
                hit_stats = rerank_analysis["hit_statistics"]
                print(f"\n  Hit queries top-1 scores ({hit_stats['count']} queries):")
                print(f"    Mean:                  {hit_stats['mean']:.4f}")
                print(f"    Median:                {hit_stats['median']:.4f}")
                print(f"    Range:                 [{hit_stats['min']:.4f}, {hit_stats['max']:.4f}]")
            
            if "miss_statistics" in rerank_analysis and rerank_analysis["miss_statistics"]:
                miss_stats = rerank_analysis["miss_statistics"]
                print(f"\n  Miss queries top-1 scores ({miss_stats['count']} queries):")
                print(f"    Mean:                  {miss_stats['mean']:.4f}")
                print(f"    Median:                {miss_stats['median']:.4f}")
                print(f"    Range:                 [{miss_stats['min']:.4f}, {miss_stats['max']:.4f}]")
            
            if rerank_analysis.get("threshold_suggestion"):
                thresh = rerank_analysis["threshold_suggestion"]
                print(f"\n  Suggested thresholds for Iteration 6:")
                print(f"    Conservative (reject low): {thresh['conservative']:.4f}")
                print(f"    Recommended:               {thresh['recommended']:.4f}")
                print(f"    Aggressive (avoid errors): {thresh['aggressive']:.4f}")
                print(f"    ({thresh['explanation']})")

    return scores, mrr_scores, results, rerank_analysis, faithfulness_analysis, relevance_analysis, validation_analysis


def main():
    """主函数：解析参数，执行完整的评估流程"""
    
    # 命令行参数解析
    ap = argparse.ArgumentParser(description="RAG 系统评估脚本（Iteration 3：chunking + 检索模式对比）")
    ap.add_argument("--corpus-dir", default="corpus", help="语料库目录路径")
    ap.add_argument("--query-file", default="corpus/queries.json", help="查询集 JSON 文件路径")
    ap.add_argument("--retrieval-top-k", type=int, default=20, help="召回的候选数量（送入 Rerank 的数量）")
    ap.add_argument("--rerank-top-k", type=int, default=5, help="Rerank 后返回的数量（送给 Generator 的数量）")
    ap.add_argument("--chunking-strategy", default="fixed_100_50", 
                    choices=["fixed_200_40", "semantic", "fixed_100_50"],
                    help="Chunking 策略（默认 fixed_100_50，Iteration 2 最优）")
    ap.add_argument("--retrieval-mode", default="hybrid",
                    choices=["random", "vector", "bm25", "hybrid"],
                    help="召回检索模式: random, vector, bm25, hybrid (默认)")
    ap.add_argument("--rerank-mode", default="none",
                    choices=["bge", "none"],
                    help="重排序模型: bge (使用 BGE Reranker), none (不使用, 默认)")
    ap.add_argument("--judge-mode", type=str, default="llm", choices=["llm", "ragas", "none"])
    ap.add_argument("--batch-size", type=int, default=10,
                    help="Judge 评估的并发批次大小（默认10，过大可能触发 API 限流）")
    ap.add_argument("--generation-version", type=str, default="v5", choices=["v5", "v6"],
                    help="生成版本: v5 (旧版本，直接标注), v6 (新版本，结构化输出+引用验证)")
    ap.add_argument("--validation-threshold", type=float, default=0.5,
                    help="V6 引用验证的词汇重叠度阈值（默认0.5）")
    args = ap.parse_args()

    # 运行单个策略和检索模式
    scores, mrr_scores, results, rerank_analysis, faithfulness_analysis, relevance_analysis, validation_analysis = run_single_strategy(args, args.chunking_strategy, args.retrieval_mode)
    
    # 保存结果（文件名包含 chunking 策略、检索模式和 rerank）
    rerank_suffix = f"_rerank_{args.rerank_mode}" if args.rerank_mode != "none" else ""
    output_file = f"results_{args.chunking_strategy}_{args.retrieval_mode}{rerank_suffix}.json"
    result_data = {
        "config": {
            "chunking_strategy": args.chunking_strategy,
            "retrieval_mode": args.retrieval_mode,
            "rerank_mode": args.rerank_mode,
            "rerank_top_k": args.rerank_top_k if args.rerank_mode != "none" else None,
            "retrieval_top_k": args.retrieval_top_k,
            "generation_version": args.generation_version,
            "validation_threshold": args.validation_threshold if args.generation_version == "v6" else None
        },
        "scores": scores,
        "mrr_scores": mrr_scores,
        "results": results
    }
    
    # 如果有 rerank 分析，也保存
    if rerank_analysis and "error" not in rerank_analysis:
        result_data["rerank_score_distribution"] = {
            "statistics": rerank_analysis["statistics"],
            "hit_statistics": rerank_analysis.get("hit_statistics"),
            "miss_statistics": rerank_analysis.get("miss_statistics"),
            "threshold_suggestion": rerank_analysis.get("threshold_suggestion")
        }
    
    # Iteration 5: 保存 Faithfulness 和 Relevance 分析
    if faithfulness_analysis:
        result_data["faithfulness_analysis"] = faithfulness_analysis
    
    if relevance_analysis:
        result_data["relevance_analysis"] = relevance_analysis
    
    # Iteration 6: 保存引用验证分析
    if validation_analysis:
        result_data["validation_analysis"] = validation_analysis
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Results written to {output_file}")



if __name__ == "__main__":
    main()
