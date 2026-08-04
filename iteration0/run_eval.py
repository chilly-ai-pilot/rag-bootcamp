"""
Iteration 0 验收运行脚本：完整的 RAG 流程

执行流程：查询 -> 检索（随机）-> 生成 -> 评分
在完整的 32 个查询集上端到端运行。分数会很差——这是预期且正确的，
对于此阶段来说。现在唯一重要的是每个步骤都能正常运行不崩溃。

使用方法:
    python run_eval.py                # 随机检索，k=5，chunk_size=200
    python run_eval.py --k 10 --chunk-size 150 --overlap 30
"""
import argparse
import json

from chunking import build_corpus_chunks
from retrieval import retrieve_random
from generation import generate_answer
from scoring import hit, aggregate_by_category


def main():
    """主函数：解析参数，执行完整的评估流程"""
    
    # 命令行参数解析
    ap = argparse.ArgumentParser(description="RAG 系统评估脚本")
    ap.add_argument("--corpus-dir", default="corpus", help="语料库目录路径")
    ap.add_argument("--queries", default="corpus/queries.json", help="查询集 JSON 文件路径")
    ap.add_argument("--chunk-size", type=int, default=200, help="文档块大小（字符数）")
    ap.add_argument("--overlap", type=int, default=40, help="相邻块的重叠字符数")
    ap.add_argument("--k", type=int, default=5, help="检索返回的块数量")
    ap.add_argument("--seed", type=int, default=42, help="随机数种子（用于可复现性）")
    ap.add_argument("--show-samples", type=int, default=3, help="打印前 N 个示例生成结果")
    ap.add_argument("--out", default="results.json", help="结果输出文件路径")
    args = ap.parse_args()

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
        # 检索相关文档块（Iteration 0 使用随机检索）
        retrieved = retrieve_random(q["query"], chunks, k=args.k, seed=args.seed + q["id"])
        
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
    print("\n=== Recall@{} by category ===".format(args.k))
    for cat, score in scores.items():
        print(f"  {cat:24s} {score:.2f}")

    # 保存完整结果到 JSON 文件
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "scores": scores, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nfull results written to {args.out}")


if __name__ == "__main__":
    main()
