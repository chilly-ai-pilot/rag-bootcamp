#!/usr/bin/env python3
"""验证所有 queries.json 中的 ground truth 位置是否正确"""

import json
import os

def verify_queries():
    """验证所有查询的 ground truth 位置"""
    
    # 加载 queries.json
    with open('corpus/queries.json', 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    print(f"验证 {len(queries)} 个查询的 ground truth 位置...\n")
    
    errors = []
    
    for q in queries:
        query_id = q['id']
        doc_id = q['doc_id']
        ground_truth = q['ground_truth_text']
        expected_start = q['char_start']
        expected_end = q['char_end']
        
        # 读取对应文档
        doc_path = f"corpus/{doc_id.replace('doc', 'doc-')}.txt"
        if not os.path.exists(doc_path):
            errors.append({
                'query_id': query_id,
                'error': f"文档不存在: {doc_path}"
            })
            continue
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            doc_text = f.read()
        
        # 验证位置
        actual_text = doc_text[expected_start:expected_end]
        
        if actual_text != ground_truth:
            # 尝试查找正确位置
            correct_pos = doc_text.find(ground_truth)
            
            if correct_pos == -1:
                errors.append({
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'error': 'ground_truth 在文档中不存在',
                    'ground_truth': ground_truth,
                    'expected_start': expected_start,
                    'expected_end': expected_end,
                    'actual_text': actual_text[:50] + '...' if len(actual_text) > 50 else actual_text
                })
            else:
                correct_end = correct_pos + len(ground_truth)
                errors.append({
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'error': '位置错误',
                    'ground_truth': ground_truth,
                    'expected_start': expected_start,
                    'expected_end': expected_end,
                    'actual_start': correct_pos,
                    'actual_end': correct_end,
                    'actual_text': actual_text[:50] + '...' if len(actual_text) > 50 else actual_text
                })
    
    # 打印结果
    if not errors:
        print("✅ 所有查询的 ground truth 位置都正确！")
        return True
    else:
        print(f"❌ 发现 {len(errors)} 个错误:\n")
        for err in errors:
            doc_id_str = f" ({err['doc_id']})" if 'doc_id' in err else ""
            print(f"Query #{err['query_id']}{doc_id_str}:")
            print(f"  错误: {err['error']}")
            if 'ground_truth' in err:
                print(f"  Ground truth: {err['ground_truth']}")
            if 'expected_start' in err:
                print(f"  当前位置: [{err['expected_start']}, {err['expected_end']}]")
            if 'actual_start' in err:
                print(f"  应该是: [{err['actual_start']}, {err['actual_end']}]")
            if 'actual_text' in err:
                print(f"  实际文本: {err['actual_text']}")
            print()
        
        return False

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    verify_queries()
