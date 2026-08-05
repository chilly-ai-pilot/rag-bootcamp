"""
Iteration 3 评估运行脚本：支持多种 chunking 策略和检索模式的对比实验

执行流程：查询 -> 检索 -> 生成 -> 评分

新增功能 (Iteration 3):
- 支持四种检索模式：random, vector, bm25, hybrid
- hybrid 模式使用 RRF（倒数排名融合）结合向量检索和 BM25

使用方法:
    # 单一策略 + 单一检索模式
    python run_eval.py --chunking-strategy small_100_50 --retrieval-mode vector
    python run_eval.py --chunking-strategy small_100_50 --retrieval-mode bm25
    python run_eval.py --chunking-strategy small_100_50 --retrieval-mode hybrid
    
    # 批量对比实验（推荐）
    python run_eval.py --compare-all    # 对比所有 chunking 策略（使用 vector 检索）
    
    # Iteration 3 典型实验：对比不同检索模式
    python run_eval.py --chunking-strategy small_100_50 --retrieval-mode vector
    python run_eval.py --chunking-strategy small_100_50 --retrieval-mode bm25
    python run_eval.py --chunking-strategy small_100_50 --retrieval-mode hybrid

提示：如果模型已在 iteration1 中下载，本脚本会自动使用缓存，无需重新下载。
如果仍看到下载信息，可以设置环境变量：export HF_HUB_OFFLINE=1
"""
import os
import argparse
import json

from chunking import build_corpus_chunks
from retrieval import retrieve_random, retrieve_vector, retrieve_bm25, retrieve_hybrid, retrieve_rerank
from generation import generate_answer
from scoring import hit, find_answer_rank, aggregate_by_category, calculate_mrr, analyze_rerank_score_distribution


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
    print(f"{'='*60}")
    
    # 加载查询集
    with open(args.queries, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # 构建文档块索引（使用指定的 chunking 策略）
    chunks = build_corpus_chunks(args.corpus_dir, strategy=chunking_strategy)
    
    # 统计块信息
    num_docs = len(set(c['doc_id'] for c in chunks))
    avg_chunk_size = sum(len(c['text']) for c in chunks) / len(chunks)
    print(f"Corpus: {len(chunks)} chunks from {num_docs} docs")
    print(f"Average chunk size: {avg_chunk_size:.1f} characters")

    # 对每个查询执行 检索 -> 生成 -> 评分 流程
    results = []
    for i, q in enumerate(queries):
        # 根据检索模式选择检索函数
        if retrieval_mode == "random":
            retrieved = retrieve_random(q["query"], chunks, k=args.k, seed=42)
        elif retrieval_mode == "vector":
            retrieved = retrieve_vector(q["query"], chunks, k=args.k, strategy=chunking_strategy)
        elif retrieval_mode == "bm25":
            retrieved = retrieve_bm25(q["query"], chunks, k=args.k)
        elif retrieval_mode == "hybrid":
            retrieved = retrieve_hybrid(q["query"], chunks, k=args.k, strategy=chunking_strategy)
        elif retrieval_mode == "rerank":
            retrieved = retrieve_rerank(q["query"], chunks, k=args.k, strategy=chunking_strategy)
        else:
            raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")
        
        # 基于检索结果生成答案
        answer = generate_answer(q["query"], retrieved)
        
        # 评估检索是否命中真实答案所在的文档块
        h = hit(retrieved, q["doc_id"], q["char_start"], q["char_end"])
        
        # 找到答案块的排名（用于 MRR 计算）
        answer_rank = find_answer_rank(retrieved, q["doc_id"], q["char_start"], q["char_end"])

        # 记录结果
        result_item = {
            "id": q["id"],
            "query": q["query"],
            "category": q["category"],
            "hit": h,
            "answer_rank": answer_rank,  # 新增：答案块排名（1-based）
            "answer": answer,
        }
        
        # 如果使用 rerank 模式，保存 rerank 分数（为 Iteration 6 做准备）
        if retrieval_mode == "rerank" and retrieved:
            result_item["rerank_scores"] = [
                c.get('rerank_score', 0.0) for c in retrieved
            ]
        
        results.append(result_item)

        # 打印前几个示例，用于人工检查
        if i < args.show_samples:
            print(f"\n--- sample {q['id']} [{q['category']}] ---")
            print("Q:", q["query"])
            print("hit:", h)
            if answer_rank:
                print(f"answer_rank: {answer_rank}")
            print("A:", answer)

    # 计算并打印分类别的 Recall@K 和 MRR 指标
    scores = aggregate_by_category(results)
    mrr_scores = calculate_mrr(results)
    
    print(f"\n=== Recall@{args.k} (chunking: {chunking_strategy}, retrieval: {retrieval_mode}) ===")
    for cat, score in scores.items():
        print(f"  {cat:24s} {score:.2f}")
    
    print(f"\n=== MRR (Mean Reciprocal Rank) ===")
    for cat, score in mrr_scores.items():
        print(f"  {cat:24s} {score:.4f}")
    
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

    return scores, mrr_scores, results, rerank_analysis


def main():
    """主函数：解析参数，执行完整的评估流程"""
    
    # 命令行参数解析
    ap = argparse.ArgumentParser(description="RAG 系统评估脚本（Iteration 3：chunking + 检索模式对比）")
    ap.add_argument("--corpus-dir", default="corpus", help="语料库目录路径")
    ap.add_argument("--queries", default="corpus/queries.json", help="查询集 JSON 文件路径")
    ap.add_argument("--k", type=int, default=5, help="检索返回的块数量")
    ap.add_argument("--chunking-strategy", default="small_100_50", 
                    choices=["fixed_200_40", "semantic", "small_100_50"],
                    help="Chunking 策略（默认 small_100_50，Iteration 2 最优）")
    ap.add_argument("--retrieval-mode", default="vector",
                    choices=["random", "vector", "bm25", "hybrid", "rerank"],
                    help="检索模式：random(Iter0), vector(Iter1/2), bm25, hybrid(Iter3), rerank(Iter4)")
    ap.add_argument("--compare-all", action="store_true",
                    help="运行所有三种 chunking 策略并生成对比结果（使用当前 retrieval-mode）")
    ap.add_argument("--show-samples", type=int, default=3, 
                    help="打印前 N 个示例生成结果（默认3，避免输出过长）")
    args = ap.parse_args()

    # 如果指定了 --compare-all，运行所有策略
    if args.compare_all:
        strategies = ["fixed_200_40", "semantic", "small_100_50"]
        all_scores = {}
        all_mrr_scores = {}
        all_results = {}
        
        for strategy in strategies:
            scores, mrr_scores, results, rerank_analysis = run_single_strategy(args, strategy)
            all_scores[strategy] = scores
            all_mrr_scores[strategy] = mrr_scores
            all_results[strategy] = results
            
            # 保存单个策略的结果（文件名包含检索模式）
            output_file = f"results_{strategy}_{args.retrieval_mode}.json"
            result_data = {
                "config": {
                    "chunking_strategy": strategy, 
                    "retrieval_mode": args.retrieval_mode,
                    "k": args.k
                },
                "scores": scores,
                "mrr_scores": mrr_scores,
                "results": results
            }
            
            # 如果有 rerank 分析，也保存
            if rerank_analysis and "error" not in rerank_analysis:
                # 不保存所有分数列表（太大），只保存统计数据
                result_data["rerank_score_distribution"] = {
                    "statistics": rerank_analysis["statistics"],
                    "hit_statistics": rerank_analysis.get("hit_statistics"),
                    "miss_statistics": rerank_analysis.get("miss_statistics"),
                    "threshold_suggestion": rerank_analysis.get("threshold_suggestion")
                }
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Results written to {output_file}")
        
        # 打印对比总结
        print(f"\n{'='*60}")
        print(f"COMPARISON SUMMARY (retrieval: {args.retrieval_mode})")
        print(f"{'='*60}")
        print(f"\nRecall@{args.k} by Strategy and Category:\n")
        
        # 获取所有类别
        categories = sorted(all_scores[strategies[0]].keys())
        
        # 打印表头
        print(f"{'Category':<25} ", end="")
        for strategy in strategies:
            print(f"{strategy:>18}", end=" ")
        print()
        print("-" * 80)
        
        # 打印每个类别的对比
        for cat in categories:
            print(f"{cat:<25} ", end="")
            for strategy in strategies:
                score = all_scores[strategy][cat]
                print(f"{score:>18.2f}", end=" ")
            print()
        
        # 打印 MRR 对比
        print(f"\nMRR (Mean Reciprocal Rank) by Strategy and Category:\n")
        print(f"{'Category':<25} ", end="")
        for strategy in strategies:
            print(f"{strategy:>18}", end=" ")
        print()
        print("-" * 80)
        
        for cat in categories:
            print(f"{cat:<25} ", end="")
            for strategy in strategies:
                score = all_mrr_scores[strategy][cat]
                print(f"{score:>18.4f}", end=" ")
            print()
        
        print(f"\n✅ Comparison complete! Check results_*_{args.retrieval_mode}.json for details.")
        
    else:
        # 运行单个策略和检索模式
        scores, mrr_scores, results, rerank_analysis = run_single_strategy(args, args.chunking_strategy, args.retrieval_mode)
        
        # 保存结果（文件名包含 chunking 策略和检索模式）
        output_file = f"results_{args.chunking_strategy}_{args.retrieval_mode}.json"
        result_data = {
            "config": {
                "chunking_strategy": args.chunking_strategy,
                "retrieval_mode": args.retrieval_mode,
                "k": args.k
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
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Results written to {output_file}")


if __name__ == "__main__":
    main()
