"""
验证并修正 citations

规则：
1. span in answer 或 sim(span, answer) ≥ 0.5
2. span in chunk 或 sim(span, chunk) ≥ 0.5
3. source 的 doc_id 与 chunk_doc_id 一致

修正策略：
- 如果 span 验证通过，但 doc_id 错误：替换为正确的 doc_id
- 如果 span 未达到 50% 阈值：标注 "[source?]" 表示不确定
- 如果 span 不在 answer 中：跳过该 citation
"""
import json
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_SIMILARITY_THRESHOLD = 0.50

def cosine_similarity(a, b):
    """计算余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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

def parse_full_id_to_source(full_id):
    """
    从 full_id 转换回 source 格式
    "doc3:chunk5" -> "文档3:片段5"
    """
    try:
        parts = full_id.replace("doc", "").replace("chunk", "").split(":")
        if len(parts) == 2:
            doc_num = int(parts[0])
            chunk_num = int(parts[1])
            return f"文档{doc_num}:片段{chunk_num}"
    except:
        pass
    return None

def correct_citations(results_file, output_file, threshold=None):
    """验证并修正 citations，生成新的 answer
    
    参数:
        results_file: 输入结果文件
        output_file: 输出文件
        threshold: 相似度阈值（默认使用 DEFAULT_SIMILARITY_THRESHOLD）
    """
    
    # 使用传入的阈值或默认值
    similarity_threshold = threshold if threshold is not None else DEFAULT_SIMILARITY_THRESHOLD
    
    # 加载结果
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    
    # 加载 embedding 模型
    print("Loading embedding model (bge-base-zh)...")
    model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
    print(f"Model loaded (threshold: {similarity_threshold})\n")
    
    stats = {
        'total_citations': 0,
        'valid': 0,
        'corrected': 0,
        'uncertain': 0,
        'skipped': 0
    }
    
    for result in results:
        query_id = result['id']
        raw_answer = result.get('raw_answer', '')
        citations = result.get('citations', [])
        
        if not citations:
            continue
        
        # 验证每个 citation 并决定如何处理
        citation_actions = []  # (pos, span, corrected_source, status)
        
        for cit in citations:
            stats['total_citations'] += 1
            
            span = cit.get("span", "")
            source = cit.get("source", "")
            chunk_text = cit.get("chunk_text", "")
            chunk_full_id = cit.get("chunk_full_id", "")  # 如 "doc3:chunk5"
            
            # 检查 span 是否在 answer 中
            if not span or span not in raw_answer:
                print(f"⚠️  Query {query_id}: Skipped (span not in answer) - {source}")
                stats['skipped'] += 1
                continue
            
            # 验证 span 是否在 chunk 中（精确或相似）
            span_in_chunk = False
            if chunk_text:
                if span in chunk_text:
                    span_in_chunk = True
                else:
                    # 计算相似度
                    span_emb = model.encode(span, normalize_embeddings=True)
                    chunk_emb = model.encode(chunk_text, normalize_embeddings=True)
                    sim_chunk = cosine_similarity(span_emb, chunk_emb)
                    
                    if sim_chunk >= similarity_threshold:
                        span_in_chunk = True
            
            # 验证 ID 是否完全匹配（doc_id + chunk_id）
            source_full_id = parse_source_to_full_id(source)
            id_match = (source_full_id == chunk_full_id) if source_full_id and chunk_full_id else False
            
            # 决定如何标注
            corrected_source = source
            status = 'valid'
            
            if span_in_chunk:
                # span 在 chunk 中，检查 ID
                if not id_match and chunk_full_id:
                    # ID 不匹配，用实际的 chunk_full_id 替换
                    corrected_source = parse_full_id_to_source(chunk_full_id)
                    if corrected_source:
                        print(f"✏️  Query {query_id}: Corrected {source} -> {corrected_source}")
                        status = 'corrected'
                        stats['corrected'] += 1
                    else:
                        # 解析失败，保持原样
                        corrected_source = source
                
                if status == 'valid':
                    stats['valid'] += 1
            else:
                # span 不在 chunk 中（未达到阈值），标记为不确定
                corrected_source = source + "?"
                print(f"❓ Query {query_id}: Uncertain {source} -> {corrected_source}")
                status = 'uncertain'
                stats['uncertain'] += 1
            
            # 记录要插入的位置和标注
            pos = raw_answer.find(span)
            if pos != -1:
                citation_actions.append((pos, span, corrected_source, status))
        
        # 按位置从后往前排序（避免位置偏移）
        citation_actions.sort(key=lambda x: x[0], reverse=True)
        
        # 重新渲染 answer
        corrected_answer = raw_answer
        for pos, span, corrected_source, status in citation_actions:
            insert_pos = pos + len(span)
            corrected_answer = corrected_answer[:insert_pos] + f"[{corrected_source}]" + corrected_answer[insert_pos:]
        
        # 更新 result
        result['answer'] = corrected_answer
        result['answer_corrected'] = True
    
    # 保存修正后的结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Citation Correction Summary")
    print(f"{'='*70}")
    print(f"Total citations:       {stats['total_citations']}")
    print(f"Valid (no change):     {stats['valid']}")
    print(f"Corrected (doc_id):    {stats['corrected']}")
    print(f"Uncertain (marked ?):  {stats['uncertain']}")
    print(f"Skipped (not in ans):  {stats['skipped']}")
    print(f"{'='*70}\n")
    print(f"✅ Corrected results saved to: {output_file}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="验证并修正 citations")
    parser.add_argument("results_file", help="输入结果 JSON 文件路径")
    parser.add_argument("output_file", nargs='?', default=None,
                       help="输出文件路径（默认在输入文件名后加 _corrected）")
    parser.add_argument("--threshold", type=float, default=None,
                       help=f"相似度阈值（0-1之间，默认 {DEFAULT_SIMILARITY_THRESHOLD}）")
    
    args = parser.parse_args()
    
    output_file = args.output_file or args.results_file.replace('.json', '_corrected.json')
    
    correct_citations(args.results_file, output_file, threshold=args.threshold)
