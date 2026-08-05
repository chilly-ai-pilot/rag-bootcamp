#!/usr/bin/env python3
"""
测试混合粒度 Hybrid Search 的效果

对比：
1. small_100_50 + vector: 0.97 (Iteration 2 最优)
2. fixed_200_40 + hybrid: 0.97 (Iteration 3 最优)
3. multi-granularity hybrid: ? (实验)
"""
import json
from retrieval import retrieve_hybrid_multi_granularity
from scoring import hit, aggregate_by_category

def main():
    # 加载查询集
    with open('corpus/queries.json') as f:
        queries = json.load(f)
    
    print("="*60)
    print("Testing Multi-Granularity Hybrid Search")
    print("  Vector: small_100_50 (short chunks, 85)")
    print("  BM25:   fixed_200_40 (long chunks, 28)")
    print("="*60)
    
    results = []
    
    for i, q in enumerate(queries):
        print(f"\n--- Query {q['id']}: {q['query'][:50]}...")
        
        # 使用混合粒度检索
        retrieved = retrieve_hybrid_multi_granularity(
            query=q['query'],
            corpus_dir='corpus',
            k=5,
            strategy_vector='small_100_50',
            strategy_bm25='fixed_200_40'
        )
        
        # 评估命中
        h = hit(retrieved, q['doc_id'], q['char_start'], q['char_end'])
        
        results.append({
            'id': q['id'],
            'query': q['query'],
            'category': q['category'],
            'hit': h,
        })
        
        print(f"  Hit: {h}")
        
        # 只显示前 3 个查询的详细信息
        if i >= 2:
            print("  (后续查询省略详细输出...)")
            # 静默模式：不打印中间信息
    
    # 计算分数
    scores = aggregate_by_category(results)
    
    print("\n" + "="*60)
    print("RESULTS: Multi-Granularity Hybrid Search")
    print("="*60)
    print(f"\nRecall@5 by category:")
    for cat, score in scores.items():
        print(f"  {cat:24s} {score:.2f}")
    
    # 对比历史最优
    print("\n" + "="*60)
    print("COMPARISON with Previous Best")
    print("="*60)
    print(f"small_100_50 + vector:       0.97  (Iteration 2)")
    print(f"fixed_200_40 + hybrid:       0.97  (Iteration 3)")
    print(f"multi-granularity hybrid:    {scores['overall']:.2f}  (Current)")
    
    if scores['overall'] > 0.97:
        print("\n🎉 SUCCESS! Multi-granularity hybrid 超过了历史最优!")
    elif scores['overall'] == 0.97:
        print("\n✅ GOOD! Multi-granularity hybrid 达到了历史最优水平!")
    else:
        print(f"\n⚠️  Multi-granularity hybrid 未超过历史最优 (差距: {0.97 - scores['overall']:.2f})")
    
    # 统计失败案例
    fails = [r for r in results if r['hit'] == 0]
    print(f"\n失败案例: {len(fails)} 个")
    for r in fails:
        print(f"  - ID {r['id']}: {r['query']} [{r['category']}]")
    
    # 保存结果
    output_file = 'results_multi_granularity.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'retrieval_mode': 'hybrid_multi_granularity',
                'vector_strategy': 'small_100_50',
                'bm25_strategy': 'fixed_200_40',
                'k': 5
            },
            'scores': scores,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")

if __name__ == "__main__":
    main()
