"""
Iteration 2 评估运行脚本：支持多种 chunking 策略的对比实验

执行流程：查询 -> 检索 -> 生成 -> 评分
支持三种 chunking 策略：fixed_200_40、semantic、small_100_50

使用方法:
    python run_eval.py --chunking-strategy fixed_200_40   # Iteration 1 baseline
    python run_eval.py --chunking-strategy semantic       # 按句子边界切分
    python run_eval.py --chunking-strategy small_100_50   # 更小粒度+更大重叠
    
    # 批量对比实验（推荐）
    python run_eval.py --compare-all    # 自动运行三种策略并生成对比结果

提示：如果模型已在 iteration1 中下载，本脚本会自动使用缓存，无需重新下载。
如果仍看到下载信息，可以设置环境变量：export HF_HUB_OFFLINE=1
"""
import os
import argparse
import json

from chunking import build_corpus_chunks
from retrieval import retrieve_random, retrieve_vector
from generation import generate_answer
from scoring import hit, aggregate_by_category


def run_single_strategy(args, chunking_strategy):
    """运行单个 chunking 策略的评估
    
    参数:
        args: 命令行参数
        chunking_strategy: chunking 策略名称
    
    返回:
        (scores, results): 分类评分和详细结果
    """
    print(f"\n{'='*60}")
    print(f"Running evaluation with chunking strategy: {chunking_strategy}")
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
        # 使用向量检索（传递 strategy 参数）
        retrieved = retrieve_vector(q["query"], chunks, k=args.k, strategy=chunking_strategy)
        
        # 基于检索结果生成答案
        answer = generate_answer(q["query"], retrieved)
        
        # 评估检索是否命中真实答案所在的文档块
        h = hit(retrieved, q["doc_id"], q["char_start"], q["char_end"])

        # 记录结果
        results.append({
            "id": q["id"],
            "query": q["query"],
            "category": q["category"],
            "hit": h,
            "answer": answer,
        })

        # 打印前几个示例，用于人工检查
        if i < args.show_samples:
            print(f"\n--- sample {q['id']} [{q['category']}] ---")
            print("Q:", q["query"])
            print("hit:", h)
            print("A:", answer)

    # 计算并打印分类别的 Recall@K 指标
    scores = aggregate_by_category(results)
    print(f"\n=== Recall@{args.k} by category (chunking: {chunking_strategy}) ===")
    for cat, score in scores.items():
        print(f"  {cat:24s} {score:.2f}")

    return scores, results


def main():
    """主函数：解析参数，执行完整的评估流程"""
    
    # 命令行参数解析
    ap = argparse.ArgumentParser(description="RAG 系统评估脚本（Iteration 2：chunking 策略对比）")
    ap.add_argument("--corpus-dir", default="corpus", help="语料库目录路径")
    ap.add_argument("--queries", default="corpus/queries.json", help="查询集 JSON 文件路径")
    ap.add_argument("--k", type=int, default=5, help="检索返回的块数量")
    ap.add_argument("--chunking-strategy", default="fixed_200_40", 
                    choices=["fixed_200_40", "semantic", "small_100_50"],
                    help="Chunking 策略")
    ap.add_argument("--compare-all", action="store_true",
                    help="运行所有三种策略并生成对比结果")
    ap.add_argument("--show-samples", type=int, default=3, 
                    help="打印前 N 个示例生成结果（默认3，避免输出过长）")
    args = ap.parse_args()

    # 如果指定了 --compare-all，运行所有策略
    if args.compare_all:
        strategies = ["fixed_200_40", "semantic", "small_100_50"]
        all_scores = {}
        all_results = {}
        
        for strategy in strategies:
            scores, results = run_single_strategy(args, strategy)
            all_scores[strategy] = scores
            all_results[strategy] = results
            
            # 保存单个策略的结果
            output_file = f"results_{strategy}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "config": {"chunking_strategy": strategy, "k": args.k},
                    "scores": scores,
                    "results": results
                }, f, ensure_ascii=False, indent=2)
            print(f"Results written to {output_file}")
        
        # 打印对比总结
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
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
        
        print(f"\n✅ Comparison complete! Check results_*.json for details.")
        
    else:
        # 运行单个策略
        scores, results = run_single_strategy(args, args.chunking_strategy)
        
        # 保存结果
        output_file = f"results_{args.chunking_strategy}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "config": {"chunking_strategy": args.chunking_strategy, "k": args.k},
                "scores": scores,
                "results": results
            }, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Results written to {output_file}")


if __name__ == "__main__":
    main()
