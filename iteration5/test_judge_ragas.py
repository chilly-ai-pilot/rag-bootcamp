"""
测试 Ragas Judge 识别错误答案的能力（异步批量版本）

验证目标：
1. Ragas Judge 能否识别 Type A 实体替换错误（目标 >80%）
2. Ragas Judge 能否识别 Type B 因果关系错误（目标 >70%）
3. Ragas Judge 能否识别 Type C 跨片段拼接错误（目标 >60%）
4. Ragas Judge 会不会误判正确答案（目标 <10%）

改进：
- 使用 asyncio 并发处理，显著提速
- 输出完整 JSON 结果，包含所有 Judge 的详细审核内容
- 处理全部 20 个错误 claims
"""

import json
import os
import asyncio
from typing import Dict, List
from evaluation import ragas_faithfulness_check_async


def load_false_claims() -> List[Dict]:
    """加载错误 Claim 测试集"""
    with open('corpus/false_claims.json', 'r', encoding='utf-8') as f:
        return json.load(f)


def load_correct_answers() -> List[Dict]:
    """从 queries.json 加载正确答案作为对照组"""
    with open('corpus/queries.json', 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    # 选取 10 个代表性的正确答案
    selected = [1, 5, 11, 15, 19, 21, 23, 25, 27, 31]
    return [q for q in queries if q['id'] in selected]


def extract_document_for_query(doc_id: str) -> str:
    """根据 doc_id 读取对应的文档内容
    
    参数:
        doc_id: 可以是 "doc1", "doc-1", "doc-1.txt" 等格式
    """
    # 处理不同格式的 doc_id
    if doc_id.endswith('.txt'):
        # 已经是完整文件名，如 "doc-1.txt"
        filename = doc_id
    elif doc_id.startswith('doc'):
        # doc1, doc-1, doc_1 等格式
        # 统一转换为 doc-X.txt
        num = doc_id.replace('doc', '').replace('_', '').replace('-', '')
        filename = f"doc-{num}.txt"
    else:
        # 其他格式，直接加 .txt
        filename = f"{doc_id}.txt"
    
    filepath = f"corpus/{filename}"
    
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return ""


async def test_ragas_connection():
    """测试 Ragas Judge 连接"""
    print("📡 测试 Ragas Judge 连接...")
    
    try:
        # 简单测试
        test_query = "SmartLock-100 使用什么电池？"
        test_answer = "使用 8 节 5 号电池"
        test_context = [{"text": "SmartLock-100 使用 8 节 5 号电池（AA），续航约 12 个月。"}]
        
        result = await ragas_faithfulness_check_async(test_query, test_answer, test_context)
        score = result.get('faithfulness_score', 0)
        
        print(f"✅ Ragas Judge 连接成功")
        print(f"📝 测试分数: {score:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Ragas Judge 连接失败: {e}")
        return False


async def test_false_claims(batch_size=10):
    """测试 Ragas Judge 对错误答案的识别能力（异步批量）"""
    print("\n" + "="*80)
    print("🔍 测试 Ragas Judge 对错误答案的识别能力")
    print(f"批次大小: {batch_size}")
    print("="*80)
    
    false_claims = load_false_claims()
    print(f"✅ 加载 {len(false_claims)} 条错误答案")
    
    results = []
    type_stats = {
        'entity_replacement': {'total': 0, 'detected': 0, 'scores': []},
        'causal_error': {'total': 0, 'detected': 0, 'scores': []},
        'cross_document': {'total': 0, 'detected': 0, 'scores': []}
    }
    
    threshold = 0.70
    
    # 分批处理
    total = len(false_claims)
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = false_claims[batch_start:batch_end]
        
        print(f"\n⏳ Processing batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch)} claims)...")
        
        # 创建异步任务（保持索引对应）
        tasks = []
        task_indices = []  # 记录哪些索引有有效任务
        
        for i, claim in enumerate(batch):
            doc_content = extract_document_for_query(claim['correct_doc'])
            if not doc_content:
                print(f"⚠️  [{claim['id']}] 无法提取文档: {claim['correct_doc']}")
                continue
            
            context = [{"text": doc_content}]
            tasks.append(
                ragas_faithfulness_check_async(
                    claim['query'],
                    claim['false_answer'],
                    context
                )
            )
            task_indices.append(i)
        
        # 并发执行
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果（使用索引对应）
        for task_idx, judge_result in enumerate(batch_results):
            batch_idx = task_indices[task_idx]
            claim = batch[batch_idx]
            
            if isinstance(judge_result, Exception):
                print(f"❌ [{claim['id']}] 评估失败: {judge_result}")
                results.append({
                    'claim_id': claim['id'],
                    'query': claim['query'],
                    'answer': claim['false_answer'],
                    'error_type': claim['error_type'],
                    'score': None,
                    'detected': False,
                    'error': str(judge_result)
                })
                continue
            
            score = judge_result.get('faithfulness_score', 0)
            detected = score < threshold
            
            # 统计
            error_type = claim['error_type']
            type_stats[error_type]['total'] += 1
            type_stats[error_type]['scores'].append(score)
            if detected:
                type_stats[error_type]['detected'] += 1
            
            # 保存结果
            results.append({
                'claim_id': claim['id'],
                'query': claim['query'],
                'answer': claim['false_answer'],
                'error_type': error_type,
                'score': score,
                'detected': detected,
                'ragas_response': judge_result.get('ragas_response', '')
            })
            
            status = "✅ 识别" if detected else "❌ 漏检"
            print(f"{status} [{claim['id']}] {claim['query'][:40]}... | Score: {score:.2f} | Type: {error_type}")
    
    return results, type_stats


async def test_correct_answers(batch_size=10):
    """测试 Ragas Judge 对正确答案的误判率（异步批量）"""
    print("\n" + "="*80)
    print("🔍 测试 Ragas Judge 对正确答案的误判率")
    print(f"批次大小: {batch_size}")
    print("="*80)
    
    queries = load_correct_answers()
    print(f"✅ 加载 {len(queries)} 条正确答案")
    
    results = []
    threshold = 0.70
    false_positives = 0
    scores = []
    
    # 分批处理
    total = len(queries)
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = queries[batch_start:batch_end]
        
        print(f"\n⏳ Processing batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch)} queries)...")
        
        # 创建异步任务（保持索引对应）
        tasks = []
        task_indices = []  # 记录哪些索引有有效任务
        
        for i, query in enumerate(batch):
            doc_content = extract_document_for_query(query['doc_id'])
            if not doc_content:
                print(f"⚠️  [Query {query['id']}] 无法提取文档: {query['doc_id']}")
                continue
            
            context = [{"text": doc_content}]
            tasks.append(
                ragas_faithfulness_check_async(
                    query['query'],
                    query['ground_truth_text'],
                    context
                )
            )
            task_indices.append(i)
        
        # 并发执行
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果（使用索引对应）
        for task_idx, judge_result in enumerate(batch_results):
            batch_idx = task_indices[task_idx]
            query = batch[batch_idx]
            
            if isinstance(judge_result, Exception):
                print(f"❌ [Query {query['id']}] 评估失败: {judge_result}")
                results.append({
                    'query_id': query['id'],
                    'query': query['query'],
                    'answer': query['ground_truth_text'],
                    'score': None,
                    'false_positive': False,
                    'error': str(judge_result)
                })
                continue
            
            score = judge_result.get('faithfulness_score', 0)
            false_positive = score < threshold
            
            if false_positive:
                false_positives += 1
            
            scores.append(score)
            
            results.append({
                'query_id': query['id'],
                'query': query['query'],
                'answer': query['ground_truth_text'],
                'score': score,
                'false_positive': false_positive,
                'ragas_response': judge_result.get('ragas_response', '')
            })
            
            status = "❌ 误判" if false_positive else "✅ 通过"
            print(f"{status} [Query {query['id']}] {query['query'][:40]}... | Score: {score:.2f}")
    
    return results, false_positives, scores


def print_summary(false_results, type_stats, correct_results, false_positives, correct_scores):
    """打印测试总结"""
    print("\n" + "="*80)
    print("📊 测试总结")
    print("="*80)
    
    print("\n【错误答案识别率】")
    for error_type, label in [
        ('entity_replacement', 'Type A - 实体替换'),
        ('causal_error', 'Type B - 因果错误'),
        ('cross_document', 'Type C - 跨片段拼接')
    ]:
        stats = type_stats[error_type]
        if stats['total'] > 0:
            rate = stats['detected'] / stats['total'] * 100
            avg_score = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
            status = "✅" if rate >= 60 else "⚠️"
            print(f"{status} {label}识别率: {stats['detected']}/{stats['total']} = {rate:.1f}%")
            print(f"   平均分数: {avg_score:.2f}")
    
    print("\n【正确答案误判率】")
    total_correct = len(correct_results)
    if total_correct > 0:
        false_positive_rate = false_positives / total_correct * 100
        avg_score = sum(correct_scores) / len(correct_scores) if correct_scores else 0
        status = "✅" if false_positive_rate < 10 else "⚠️"
        print(f"{status} 误判率: {false_positives}/{total_correct} = {false_positive_rate:.1f}%")
        print(f"   平均分数: {avg_score:.2f}")
    
    print("\n【总体评估】")
    all_detected = sum(s['detected'] for s in type_stats.values())
    all_total = sum(s['total'] for s in type_stats.values())
    overall_rate = all_detected / all_total * 100 if all_total > 0 else 0
    
    if overall_rate >= 70 and false_positive_rate < 10:
        print("🎉 Ragas Judge 通过测试！可以用于评估。")
    else:
        print("⚠️  Ragas Judge 性能不足，建议使用 LLM Judge。")


def save_results(false_results, correct_results):
    """保存详细测试结果"""
    output = {
        'test_type': 'ragas_judge',
        'threshold': 0.70,
        'false_claims': false_results,
        'correct_answers': correct_results
    }
    
    output_file = 'judge_test_results_ragas.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细测试结果已保存到 {output_file}")


async def main():
    print("🚀 开始测试 Ragas Judge 模型能力（异步批量版本）")
    print("="*80)
    
    # 测试连接
    if not await test_ragas_connection():
        print("\n❌ 测试失败")
        return
    
    # 测试错误答案识别
    false_results, type_stats = await test_false_claims(batch_size=10)
    
    # 测试正确答案误判率
    correct_results, false_positives, correct_scores = await test_correct_answers(batch_size=10)
    
    # 打印总结
    print_summary(false_results, type_stats, correct_results, false_positives, correct_scores)
    
    # 保存结果
    save_results(false_results, correct_results)
    
    print("\n" + "="*80)
    print("✅ 所有测试完成！")


if __name__ == "__main__":
    asyncio.run(main())
