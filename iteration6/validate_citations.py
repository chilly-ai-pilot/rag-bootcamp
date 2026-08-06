"""
验证 citations 质量，检测 citation hallucination

检查：
1. span 是否在 raw_answer 中（精确匹配）
2. 统计 hallucination 率
"""
import json
import sys

def validate_citations(results_file):
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    
    total_citations = 0
    hallucinated_citations = 0
    hallucinated_examples = []
    
    print(f"\n{'='*60}")
    print(f"Citation Validation Report")
    print(f"{'='*60}\n")
    
    for result in results:
        query_id = result['id']
        query_text = result['query']
        raw_answer = result['raw_answer']
        citations = result['citations']
        
        for cit in citations:
            total_citations += 1
            span = cit['span']
            source = cit['source']
            
            # 检查 span 是否在 raw_answer 中
            if span not in raw_answer:
                hallucinated_citations += 1
                hallucinated_examples.append({
                    'query_id': query_id,
                    'query': query_text,
                    'span': span,
                    'source': source,
                    'raw_answer': raw_answer
                })
                print(f"❌ Query {query_id}: Citation hallucination detected!")
                print(f"   Span: '{span}'")
                print(f"   Source: {source}")
                print(f"   Raw answer: '{raw_answer[:100]}...'")
                print()
    
    # 统计
    hallucination_rate = (hallucinated_citations / total_citations * 100) if total_citations > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Statistics")
    print(f"{'='*60}")
    print(f"Total citations:       {total_citations}")
    print(f"Hallucinated:          {hallucinated_citations} ({hallucination_rate:.1f}%)")
    print(f"Valid:                 {total_citations - hallucinated_citations} ({100 - hallucination_rate:.1f}%)")
    print(f"{'='*60}\n")
    
    if hallucination_rate == 0:
        print("✅ 完美！没有检测到 citation hallucination！")
        print("   两步法成功解决了 LLM 引用不存在内容的问题。\n")
    else:
        print(f"⚠️  仍有 {hallucination_rate:.1f}% 的 citations 存在 hallucination")
        print(f"   需要进一步优化 prompt 或验证逻辑。\n")
    
    return {
        'total': total_citations,
        'hallucinated': hallucinated_citations,
        'rate': hallucination_rate,
        'examples': hallucinated_examples
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate_citations.py <results_file.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    validate_citations(results_file)
