"""
Iteration 1 评估运行脚本：支持多种检索策略的 RAG 流程

执行流程：查询 -> 检索（random/vector）-> 生成 -> 评分
支持通过 --strategy 参数选择检索策略。

使用方法:
    python run_eval.py --strategy random   # Iteration 0 基线（随机检索）
    python run_eval.py --strategy vector   # Iteration 1（向量检索）
    python run_eval.py --strategy vector --k 10 --chunk-size 150
"""
import argparse
import json

from chunking import build_corpus_chunks
from retrieval import retrieve_random, retrieve_vector
from generation import generate_answer
from scoring import hit, aggregate_by_category


def main():
    """主函数：解析参数，执行完整的评估流程"""
    
    # 命令行参数解析
    ap = argparse.ArgumentParser(description="RAG 系统评估脚本（支持多种检索策略）")
    ap.add_argument("--corpus-dir", default="corpus", help="语料库目录路径")
    ap.add_argument("--queries", default="corpus/queries.json", help="查询集 JSON 文件路径")
    ap.add_argument("--chunk-size", type=int, default=200, help="文档块大小（字符数）")
    ap.add_argument("--overlap", type=int, default=40, help="相邻块的重叠字符数")
    ap.add_argument("--k", type=int, default=5, help="检索返回的块数量")
    ap.add_argument("--strategy", default="random", choices=["random", "vector"], 
                    help="检索策略：random（Iteration 0）或 vector（Iteration 1）")
    ap.add_argument("--seed", type=int, default=42, help="随机数种子（仅用于 random 策略）")
    ap.add_argument("--show-samples", type=int, default=32, help="打印前 N 个示例生成结果")
    ap.add_argument("--out", default="results.json", help="结果输出文件路径")
    args = ap.parse_args()

    # 选择检索函数
    if args.strategy == "random":
        retrieve_fn = retrieve_random
        print(f"Using RANDOM retrieval strategy (Iteration 0 baseline)")
    elif args.strategy == "vector":
        retrieve_fn = retrieve_vector
        print(f"Using VECTOR retrieval strategy (Iteration 1: bge-base-zh + ChromaDB)")
    
    # 加载查询集
    with open(args.queries, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # 构建文档块索引
    chunks = build_corpus_chunks(args.corpus_dir, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"corpus: {len(chunks)} chunks from {len(set(c['doc_id'] for c in chunks))} docs "
          f"(chunk_size={args.chunk_size}, overlap={args.overlap})")

    # 对每个查询执行 检索 -> 生成 -> 评分 流程
    results = []
    for i, q in enumerate(queries):
        # 检索相关文档块（根据 strategy 参数选择策略）
        if args.strategy == "random":
            retrieved = retrieve_fn(q["query"], chunks, k=args.k, seed=args.seed + q["id"])
        else:
            retrieved = retrieve_fn(q["query"], chunks, k=args.k)
        
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
    print("\n=== Recall@{} by category (strategy: {}) ===".format(args.k, args.strategy))
    for cat, score in scores.items():
        print(f"  {cat:24s} {score:.2f}")

    # 保存完整结果到 JSON 文件
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "scores": scores, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nfull results written to {args.out}")


if __name__ == "__main__":
    main()
