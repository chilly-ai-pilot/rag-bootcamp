#!/usr/bin/env python3
"""测试不同检索模式对失败查询的效果"""

import json
from chunking import build_corpus_chunks
from retrieval import retrieve_vector, retrieve_bm25, retrieve_hybrid, rerank_chunks

# 失败的查询
FAILED_QUERIES = [
    {"id": 7, "query": "SmartPlug-400 的充电保护功能是怎样工作的？", "doc_id": "doc6"},
    {"id": 18, "query": "SmartPlug-400 待机功耗多少瓦？", "doc_id": "doc6"},
    {"id": 22, "query": "怎样防止摄像头被偷看？", "doc_id": "doc2"},
    {"id": 27, "query": "传感器没电了怎么换？", "doc_id": "doc5"},
    {"id": 31, "query": "SW-600 的电池型号是什么？", "doc_id": "doc7"},
    {"id": 32, "query": "场景面板的按键图标能不能自定义？", "doc_id": "doc7"},
]

# 加载 queries.json 获取 ground truth
with open('corpus/queries.json', 'r', encoding='utf-8') as f:
    all_queries = json.load(f)
    
query_gt_map = {}
for q in all_queries:
    query_gt_map[q['id']] = {
        'char_start': q['char_start'],
        'char_end': q['char_end'],
        'ground_truth_text': q['ground_truth_text']
    }

def check_hit(retrieved, doc_id, char_start, char_end):
    """检查是否命中 ground truth"""
    for chunk in retrieved:
        if chunk['doc_id'] != doc_id:
            continue
        
        # 读取原文档
        doc_path = f"corpus/{doc_id.replace('doc', 'doc-')}.txt"
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_text = f.read()
        
        # 检查 chunk 是否包含 ground truth
        gt_text = doc_text[char_start:char_end]
        if gt_text in chunk['text']:
            return True
    return False

def find_rank(retrieved, doc_id, char_start, char_end):
    """找到 ground truth 的排名"""
    doc_path = f"corpus/{doc_id.replace('doc', 'doc-')}.txt"
    with open(doc_path, 'r', encoding='utf-8') as f:
        doc_text = f.read()
    
    gt_text = doc_text[char_start:char_end]
    
    for i, chunk in enumerate(retrieved):
        if chunk['doc_id'] == doc_id and gt_text in chunk['text']:
            return i + 1
    return None

# 构建 chunks
print("构建 corpus chunks...")
chunks = build_corpus_chunks('corpus', strategy='fixed_100_50')
print(f"总共 {len(chunks)} 个 chunks\n")

# 测试每个查询
results = []

for fq in FAILED_QUERIES:
    query_id = fq['id']
    query = fq['query']
    doc_id = fq['doc_id']
    
    gt = query_gt_map[query_id]
    char_start = gt['char_start']
    char_end = gt['char_end']
    
    print(f"{'='*80}")
    print(f"Query #{query_id}: {query}")
    print(f"Expected doc: {doc_id}")
    print(f"Ground truth: {gt['ground_truth_text'][:50]}...")
    print(f"{'='*80}")
    
    result = {
        'id': query_id,
        'query': query,
        'doc_id': doc_id,
        'modes': {}
    }
    
    # 1. Vector only (top-40)
    print("\n[1] Vector (top-40):")
    retrieved_vector = retrieve_vector(query, chunks, k=40, strategy='fixed_100_50')
    hit_vector = check_hit(retrieved_vector, doc_id, char_start, char_end)
    rank_vector = find_rank(retrieved_vector, doc_id, char_start, char_end)
    print(f"    Hit: {hit_vector}, Rank: {rank_vector if rank_vector else 'N/A'}")
    result['modes']['vector'] = {'hit': hit_vector, 'rank': rank_vector}
    
    # 2. BM25 only (top-40)
    print("\n[2] BM25 (top-40):")
    retrieved_bm25 = retrieve_bm25(query, chunks, k=40)
    hit_bm25 = check_hit(retrieved_bm25, doc_id, char_start, char_end)
    rank_bm25 = find_rank(retrieved_bm25, doc_id, char_start, char_end)
    print(f"    Hit: {hit_bm25}, Rank: {rank_bm25 if rank_bm25 else 'N/A'}")
    result['modes']['bm25'] = {'hit': hit_bm25, 'rank': rank_bm25}
    
    # 3. Hybrid (top-40)
    print("\n[3] Hybrid (top-40):")
    retrieved_hybrid = retrieve_hybrid(query, chunks, k=40, strategy='fixed_100_50')
    hit_hybrid = check_hit(retrieved_hybrid, doc_id, char_start, char_end)
    rank_hybrid = find_rank(retrieved_hybrid, doc_id, char_start, char_end)
    print(f"    Hit: {hit_hybrid}, Rank: {rank_hybrid if rank_hybrid else 'N/A'}")
    result['modes']['hybrid'] = {'hit': hit_hybrid, 'rank': rank_hybrid}
    
    # 4. Vector + Rerank (top-40)
    print("\n[4] Vector + Rerank BGE (top-40):")
    reranked_vector = rerank_chunks(query, retrieved_vector, top_k=40)
    hit_vector_rerank = check_hit(reranked_vector, doc_id, char_start, char_end)
    rank_vector_rerank = find_rank(reranked_vector, doc_id, char_start, char_end)
    print(f"    Hit: {hit_vector_rerank}, Rank: {rank_vector_rerank if rank_vector_rerank else 'N/A'}")
    result['modes']['vector_rerank'] = {'hit': hit_vector_rerank, 'rank': rank_vector_rerank}
    
    # 5. Hybrid + Rerank (top-40)
    print("\n[5] Hybrid + Rerank BGE (top-40):")
    reranked_hybrid = rerank_chunks(query, retrieved_hybrid, top_k=40)
    hit_hybrid_rerank = check_hit(reranked_hybrid, doc_id, char_start, char_end)
    rank_hybrid_rerank = find_rank(reranked_hybrid, doc_id, char_start, char_end)
    print(f"    Hit: {hit_hybrid_rerank}, Rank: {rank_hybrid_rerank if rank_hybrid_rerank else 'N/A'}")
    result['modes']['hybrid_rerank'] = {'hit': hit_hybrid_rerank, 'rank': rank_hybrid_rerank}
    
    # 6. BM25 + Rerank (top-40)
    print("\n[6] BM25 + Rerank BGE (top-40):")
    reranked_bm25 = rerank_chunks(query, retrieved_bm25, top_k=40)
    hit_bm25_rerank = check_hit(reranked_bm25, doc_id, char_start, char_end)
    rank_bm25_rerank = find_rank(reranked_bm25, doc_id, char_start, char_end)
    print(f"    Hit: {hit_bm25_rerank}, Rank: {rank_bm25_rerank if rank_bm25_rerank else 'N/A'}")
    result['modes']['bm25_rerank'] = {'hit': hit_bm25_rerank, 'rank': rank_bm25_rerank}
    
    results.append(result)
    print()

# 汇总统计
print(f"\n{'='*80}")
print("汇总统计")
print(f"{'='*80}\n")

modes = ['vector', 'bm25', 'hybrid', 'vector_rerank', 'bm25_rerank', 'hybrid_rerank']
mode_names = {
    'vector': 'Vector only',
    'bm25': 'BM25 only',
    'hybrid': 'Hybrid',
    'vector_rerank': 'Vector+Rerank',
    'bm25_rerank': 'BM25+Rerank',
    'hybrid_rerank': 'Hybrid+Rerank'
}

for mode in modes:
    hits = sum(1 for r in results if r['modes'][mode]['hit'])
    print(f"{mode_names[mode]:20s}: {hits}/6 命中")

# 详细表格
print(f"\n{'='*80}")
print("详细结果表格")
print(f"{'='*80}\n")
print(f"{'Query':<6} {'Vector':<10} {'BM25':<10} {'Hybrid':<10} {'V+Rerank':<12} {'B+Rerank':<12} {'H+Rerank':<12}")
print(f"{'-'*80}")

for r in results:
    qid = f"Q{r['id']}"
    
    def format_result(mode_data):
        if mode_data['hit']:
            return f"✅ R{mode_data['rank']}"
        else:
            return "❌"
    
    print(f"{qid:<6} "
          f"{format_result(r['modes']['vector']):<10} "
          f"{format_result(r['modes']['bm25']):<10} "
          f"{format_result(r['modes']['hybrid']):<10} "
          f"{format_result(r['modes']['vector_rerank']):<12} "
          f"{format_result(r['modes']['bm25_rerank']):<12} "
          f"{format_result(r['modes']['hybrid_rerank']):<12}")

print(f"\n✅ = 命中, R = 排名")
