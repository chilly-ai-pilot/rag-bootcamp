"""
验证 citation attribution（引用归因）

验证规则：
1. span in answer 或 similarity(span, answer) ≥ 0.5
2. span in chunk 或 similarity(span, chunk) ≥ 0.5
3. source 指向的 doc_id 与 chunk 的 doc_id 一致

同时满足才算通过
"""
import json
import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# 默认相似度阈值
DEFAULT_SIMILARITY_THRESHOLD = 0.50

def cosine_similarity(a, b):
    """计算余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_chunks(corpus_dir, chunking_strategy):
    """加载所有文档块（复用 chunking.py 的逻辑）"""
    from chunking import build_corpus_chunks
    return build_corpus_chunks(corpus_dir, strategy=chunking_strategy)

def parse_source_to_full_id(source):
    """
    从 source 转换为 full_id 格式
    "文档3:片段5" -> "doc3:chunk5"
    """
    try:
        parts = source.replace("文档", "").replace("片段", "").split(":")
        if len(parts) == 2:
            doc_num = int(parts[0])
            chunk_num = int(parts[1])
            return f"doc{doc_num}:chunk{chunk_num}"
    except:
        pass
    return None

def validate_attribution(results_file, corpus_dir=None, chunking_strategy=None, threshold=None):
    """验证 citation attribution
    
    参数:
        results_file: 结果文件路径
        corpus_dir: 语料库目录（可选）
        chunking_strategy: 分块策略（可选）
        threshold: 相似度阈值（默认使用 DEFAULT_SIMILARITY_THRESHOLD）
    """
    
    # 使用传入的阈值或默认值
    similarity_threshold = threshold if threshold is not None else DEFAULT_SIMILARITY_THRESHOLD
    
    # 加载结果
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    
    # 加载 embedding 模型
    print("\nLoading embedding model (bge-base-zh)...")
    model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
    print("Model loaded")
    
    print(f"\n{'='*70}")
    print(f"Citation Attribution Validation (Threshold: {similarity_threshold})")
    print(f"{'='*70}\n")
    
    total_citations = 0
    valid_citations = 0
    invalid_citations = 0
    no_chunk_text = 0
    failed_examples = []
    
    # 统计细分
    stats = {
        'answer_exact': 0,
        'answer_similar': 0,
        'answer_fail': 0,
        'chunk_exact': 0,
        'chunk_similar': 0,
        'chunk_fail': 0,
        'doc_id_match': 0,
        'doc_id_mismatch': 0
    }
    
    for result in results:
        query_id = result['id']
        query_text = result['query']
        raw_answer = result.get('raw_answer', '')
        citations = result.get('citations', [])
        
        for cit in citations:
            total_citations += 1
            span = cit['span']
            source = cit['source']
            chunk_text = cit.get('chunk_text')
            chunk_full_id = cit.get('chunk_full_id')  # 如 "doc3:chunk5"
            
            # 检查是否有 chunk_text
            if not chunk_text:
                print(f"⚠️  Query {query_id}: No chunk_text for source '{source}'")
                no_chunk_text += 1
                invalid_citations += 1
                failed_examples.append({
                    'query_id': query_id,
                    'span': span,
                    'source': source,
                    'reason': 'no_chunk_text',
                })
                continue
            
            # ========================================
            # 验证 1: span 是否在 answer 中（精确或相似）
            # ========================================
            answer_valid = False
            answer_method = None
            
            if span in raw_answer:
                answer_valid = True
                answer_method = 'exact'
                stats['answer_exact'] += 1
            else:
                # 计算相似度
                span_emb = model.encode(span, normalize_embeddings=True)
                answer_emb = model.encode(raw_answer, normalize_embeddings=True)
                sim_answer = cosine_similarity(span_emb, answer_emb)
                
                if sim_answer >= similarity_threshold:
                    answer_valid = True
                    answer_method = f'similar({sim_answer:.3f})'
                    stats['answer_similar'] += 1
                else:
                    answer_method = f'fail({sim_answer:.3f})'
                    stats['answer_fail'] += 1
            
            # ========================================
            # 验证 2: span 是否在 chunk 中（精确或相似）
            # ========================================
            chunk_valid = False
            chunk_method = None
            
            if span in chunk_text:
                chunk_valid = True
                chunk_method = 'exact'
                stats['chunk_exact'] += 1
            else:
                # 计算相似度
                span_emb = model.encode(span, normalize_embeddings=True)
                chunk_emb = model.encode(chunk_text, normalize_embeddings=True)
                sim_chunk = cosine_similarity(span_emb, chunk_emb)
                
                if sim_chunk >= similarity_threshold:
                    chunk_valid = True
                    chunk_method = f'similar({sim_chunk:.3f})'
                    stats['chunk_similar'] += 1
                else:
                    chunk_method = f'fail({sim_chunk:.3f})'
                    stats['chunk_fail'] += 1
            
            # ========================================
            # 验证 3: source 是否与 chunk_full_id 完全匹配
            # "文档3:片段5" vs "doc3:chunk5"
            # ========================================
            source_full_id = parse_source_to_full_id(source)
            id_valid = (source_full_id == chunk_full_id) if source_full_id and chunk_full_id else False
            
            if id_valid:
                stats['doc_id_match'] += 1
            else:
                stats['doc_id_mismatch'] += 1
            
            # ========================================
            # 综合判断：三个条件都满足才通过
            # ========================================
            if answer_valid and chunk_valid and id_valid:
                valid_citations += 1
                print(f"✅ Query {query_id}: {source}")
                print(f"   Answer: {answer_method}, Chunk: {chunk_method}, ID: match")
                print(f"   Span: '{span[:60]}...'")
            else:
                invalid_citations += 1
                reasons = []
                if not answer_valid:
                    reasons.append(f"answer_{answer_method}")
                if not chunk_valid:
                    reasons.append(f"chunk_{chunk_method}")
                if not id_valid:
                    reasons.append(f"id_mismatch(expected={source_full_id}, got={chunk_full_id})")
                
                reason_str = ", ".join(reasons)
                print(f"❌ Query {query_id}: {source}")
                print(f"   Failed: {reason_str}")
                print(f"   Span: '{span[:60]}...'")
                print()
                
                failed_examples.append({
                    'query_id': query_id,
                    'query': query_text,
                    'span': span,
                    'source': source,
                    'reason': reason_str,
                    'answer_valid': answer_valid,
                    'chunk_valid': chunk_valid,
                    'id_valid': id_valid,
                })
    
    # 统计
    accuracy = (valid_citations / total_citations * 100) if total_citations > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"Statistics")
    print(f"{'='*70}")
    print(f"Total citations:       {total_citations}")
    print(f"Valid (all 3 checks):  {valid_citations} ({accuracy:.1f}%)")
    print(f"Invalid:               {invalid_citations} ({100-accuracy:.1f}%)")
    if no_chunk_text > 0:
        print(f"  - No chunk_text:     {no_chunk_text}")
        print(f"  - Failed validation: {invalid_citations - no_chunk_text}")
    print(f"\nValidation Breakdown:")
    print(f"  Answer verification:")
    print(f"    Exact match:       {stats['answer_exact']}")
    print(f"    Similar (≥{similarity_threshold}):      {stats['answer_similar']}")
    print(f"    Failed (<{similarity_threshold}):       {stats['answer_fail']}")
    print(f"  Chunk verification:")
    print(f"    Exact match:       {stats['chunk_exact']}")
    print(f"    Similar (≥{similarity_threshold}):      {stats['chunk_similar']}")
    print(f"    Failed (<{similarity_threshold}):       {stats['chunk_fail']}")
    print(f"  Doc/Chunk ID verification:")
    print(f"    Match:             {stats['doc_id_match']}")
    print(f"    Mismatch:          {stats['doc_id_mismatch']}")
    print(f"{'='*70}\n")
    
    if accuracy >= 90:
        print(f"✅ 优秀！{accuracy:.1f}% 的 citations 语义相似度 ≥ {similarity_threshold}")
    elif accuracy >= 80:
        print(f"⚠️  良好，但有 {100-accuracy:.1f}% 的 citations 相似度偏低")
    else:
        print(f"❌ 需要改进：{100-accuracy:.1f}% 的 citations 与 source 语义不匹配")
    
    # 打印失败案例详情
    if failed_examples:
        print(f"\n{'='*70}")
        print(f"Failed Citations Details (first 5)")
        print(f"{'='*70}\n")
        for i, ex in enumerate(failed_examples[:5], 1):
            print(f"Example {i}:")
            print(f"  Query {ex['query_id']}: {ex.get('query', 'N/A')}")
            print(f"  Source: {ex['source']}")
            print(f"  Reason: {ex['reason']}")
            if ex.get('similarity') is not None:
                print(f"  Similarity: {ex['similarity']:.3f}")
            print(f"  Span: '{ex['span'][:100]}'")
            if ex.get('chunk_text'):
                print(f"  Chunk: '{ex['chunk_text'][:150]}...'")
            print()
    
    return {
        'total': total_citations,
        'valid': valid_citations,
        'invalid': invalid_citations,
        'accuracy': accuracy,
        'failed_examples': failed_examples
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="验证 citation attribution")
    parser.add_argument("results_file", help="结果 JSON 文件路径")
    parser.add_argument("--threshold", type=float, default=None,
                       help=f"相似度阈值（0-1之间，默认 {DEFAULT_SIMILARITY_THRESHOLD}）")
    parser.add_argument("--corpus-dir", default=None, help="语料库目录（可选）")
    parser.add_argument("--chunking-strategy", default=None, help="分块策略（可选）")
    
    args = parser.parse_args()
    
    validate_attribution(
        args.results_file,
        corpus_dir=args.corpus_dir,
        chunking_strategy=args.chunking_strategy,
        threshold=args.threshold
    )
