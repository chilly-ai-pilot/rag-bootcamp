"""
测试 LLM LLM Judge 模型识别错误答案的能力（异步批量版本）

验证目标：
1. LLM Judge 能否识别 Type A 实体替换错误（目标 >80%）
2. LLM Judge 能否识别 Type B 因果关系错误（目标 >70%）
3. LLM Judge 能否识别 Type C 跨片段拼接错误（目标 >60%）
4. LLM Judge 会不会误判正确答案（目标 <10%）

改进：
- 使用 asyncio 并发处理，显著提速
- 输出完整 JSON 结果，包含所有 Judge 的详细审核内容
- 处理全部 20 个错误 claims
"""

import json
import os
import asyncio
import re
from typing import Dict, List
from openai import AsyncOpenAI
from evaluation import llm_faithfulness_check_async


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
    """根据 doc_id 读取对应的文档内容"""
    doc_map = {
        "doc1": "doc-1.txt",
        "doc2": "doc-2.txt",
        "doc3": "doc-3.txt",
        "doc4": "doc-4.txt",
        "doc5": "doc-5.txt",
        "doc6": "doc-6.txt",
        "doc7": "doc-7.txt",
    }
    
    filename = doc_map.get(doc_id, f"{doc_id}.txt")
    filepath = f"corpus/{filename}"
    
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return ""


def parse_faithfulness_score(response: str) -> float:
    """从 Judge 的响应中提取 Faithfulness 分数"""
    # 尝试多种模式匹配分数
    patterns = [
        r'【Faithfulness 分数】\s*\n?\s*([0-9.]+)',
        r'Faithfulness.*?分数.*?[:：]\s*([0-9.]+)',
        r'最终分数[:：]\s*([0-9.]+)',
        r'总分[:：]\s*([0-9.]+)',
        r'([0-9.]+)\s*/\s*1\.0',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                score = float(match.group(1))
                # 如果分数大于1，可能是百分制，转换为0-1
                if score > 1:
                    score = score / 100
                return score
            except:
                continue
    
    # 如果无法提取，找所有数字
    all_scores = re.findall(r'\b([0-9]\.[0-9]{1,2})\b', response)
    if all_scores:
        return float(all_scores[-1])
    
    # 如果无法提取，使用关键词判断
    if any(word in response for word in ['完全编造', '无依据', '错误', '不符']):
        return 0.3
    elif any(word in response for word in ['部分正确', '部分依据']):
        return 0.5
    elif any(word in response for word in ['完全正确', '都有依据', '忠实']):
        return 0.9
    
    # 默认返回中等分数
    return 0.5


async def test_false_claims_async(client: AsyncOpenAI, claims: List[Dict], batch_size: int = 10) -> Dict:
    """异步批量测试 LLM Judge 对错误答案的识别能力
    
    参数:
        client: AsyncOpenAI 客户端
        claims: 错误 claim 列表
        batch_size: 批次大小（默认 10）
    
    返回:
        包含详细结果的字典
    """
    results = {
        'entity_replacement': {'total': 0, 'detected': 0, 'scores': [], 'details': []},
        'causal_error': {'total': 0, 'detected': 0, 'scores': [], 'details': []},
        'cross_document': {'total': 0, 'detected': 0, 'scores': [], 'details': []},
    }
    
    print("🔍 测试 LLM Judge 对错误答案的识别能力")
    print(f"   批次大小: {batch_size}")
    print("=" * 80)
    
    # 分批处理
    total = len(claims)
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_claims = claims[batch_start:batch_end]
        
        print(f"\n⏳ Processing batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch_claims)} claims)...")
        
        # 准备当前批次的任务
        tasks = []
        for claim in batch_claims:
            # 获取文档内容
            doc_id = claim['correct_doc'].replace('.txt', '').replace('-', '')
            doc_content = extract_document_for_query(doc_id)
            
            if not doc_content:
                print(f"⚠️  警告：无法读取文档 {claim['correct_doc']}")
                continue
            
            # 构建 chunk 格式（和 evaluation.py 一致）
            chunk = {
                'doc_id': doc_id,
                'text': doc_content,
                'start': 0,
                'end': len(doc_content)
            }
            
            tasks.append({
                'claim': claim,
                'coro': llm_faithfulness_check_async(
                    claim['false_answer'],
                    [chunk],
                    client
                )
            })
        
        # 并发执行当前批次
        batch_results = await asyncio.gather(*[t['coro'] for t in tasks], return_exceptions=True)
        
        # 处理结果
        for task_info, judge_result in zip(tasks, batch_results):
            claim = task_info['claim']
            error_type = claim['error_type']
            results[error_type]['total'] += 1
            
            if isinstance(judge_result, Exception):
                print(f"❌ Claim {claim['id']} 失败: {judge_result}")
                continue
            
            # 提取分数
            score = parse_faithfulness_score(judge_result['raw_response'])
            results[error_type]['scores'].append(score)
            
            # 判断是否成功识别错误（分数 < 0.7 视为识别出错误）
            detected = score < 0.7
            if detected:
                results[error_type]['detected'] += 1
            
            # 保存详细信息
            detail = {
                'id': claim['id'],
                'query': claim['query'],
                'error_type': error_type,
                'difficulty': claim['difficulty'],
                'error_description': claim['error_description'],
                'correct_answer': claim['correct_answer'],
                'false_answer': claim['false_answer'],
                'faithfulness_score': score,
                'detected': detected,
                'judge_response': judge_result['raw_response'],
                'judge_prompt': judge_result['prompt']
            }
            results[error_type]['details'].append(detail)
            
            # 简化打印
            status = "✅" if detected else "❌"
            print(f"   {status} [{claim['id']}] {claim['query'][:40]}... | Score: {score:.2f} | Type: {error_type}")
    
    return results


async def test_correct_answers_async(client: AsyncOpenAI, queries: List[Dict], batch_size: int = 10) -> Dict:
    """异步批量测试 LLM Judge 对正确答案的误判率
    
    参数:
        client: AsyncOpenAI 客户端
        queries: 正确查询列表
        batch_size: 批次大小（默认 10）
    
    返回:
        包含详细结果的字典
    """
    results = {
        'total': 0,
        'false_positive': 0,
        'scores': [],
        'details': []
    }
    
    print("\n\n" + "=" * 80)
    print("🔍 测试 LLM Judge 对正确答案的误判率")
    print(f"   批次大小: {batch_size}")
    print("=" * 80)
    
    # 分批处理
    total = len(queries)
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_queries = queries[batch_start:batch_end]
        
        print(f"\n⏳ Processing batch {batch_start//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch_queries)} queries)...")
        
        # 准备当前批次的任务
        tasks = []
        for query in batch_queries:
            doc_id = query.get('doc_id', '').replace('-', '')
            doc_content = extract_document_for_query(doc_id)
            
            if not doc_content:
                continue
            
            # 构建 chunk 格式
            chunk = {
                'doc_id': doc_id,
                'text': doc_content,
                'start': 0,
                'end': len(doc_content)
            }
            
            # 使用 ground_truth_text 作为正确答案
            correct_answer = query['ground_truth_text']
            
            tasks.append({
                'query': query,
                'coro': llm_faithfulness_check_async(
                    correct_answer,
                    [chunk],
                    client
                )
            })
        
        # 并发执行当前批次
        batch_results = await asyncio.gather(*[t['coro'] for t in tasks], return_exceptions=True)
        
        # 处理结果
        for task_info, judge_result in zip(tasks, batch_results):
            query = task_info['query']
            results['total'] += 1
            
            if isinstance(judge_result, Exception):
                print(f"❌ Query {query['id']} 失败: {judge_result}")
                continue
            
            # 提取分数
            score = parse_faithfulness_score(judge_result['raw_response'])
            results['scores'].append(score)
            
            # 判断是否误判（分数 < 0.7 视为误判为错误）
            false_positive = score < 0.7
            if false_positive:
                results['false_positive'] += 1
            
            # 保存详细信息
            detail = {
                'id': query['id'],
                'query': query['query'],
                'category': query['category'],
                'correct_answer': query['ground_truth_text'],
                'faithfulness_score': score,
                'false_positive': false_positive,
                'judge_response': judge_result['raw_response'],
                'judge_prompt': judge_result['prompt']
            }
            results['details'].append(detail)
            
            # 简化打印
            status = "✅" if not false_positive else "❌"
            print(f"   {status} [Query {query['id']}] {query['query'][:40]}... | Score: {score:.2f}")
    
    return results


def print_summary(false_results: Dict, correct_results: Dict):
    """打印测试总结"""
    print("\n\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)
    
    print("\n【错误答案识别率】")
    for error_type, data in false_results.items():
        if data['total'] > 0:
            accuracy = data['detected'] / data['total'] * 100
            avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
            
            type_name = {
                'entity_replacement': 'Type A - 实体替换',
                'causal_error': 'Type B - 因果错误',
                'cross_document': 'Type C - 跨片段拼接'
            }[error_type]
            
            target = {
                'entity_replacement': 80,
                'causal_error': 70,
                'cross_document': 60
            }[error_type]
            
            status = "✅" if accuracy >= target else "⚠️"
            
            print(f"\n{status} {type_name}")
            print(f"   识别率: {data['detected']}/{data['total']} = {accuracy:.1f}% (目标: {target}%)")
            print(f"   平均分数: {avg_score:.2f}")
    
    print("\n【正确答案误判率】")
    if correct_results['total'] > 0:
        false_positive_rate = correct_results['false_positive'] / correct_results['total'] * 100
        avg_score = sum(correct_results['scores']) / len(correct_results['scores']) if correct_results['scores'] else 0
        
        status = "✅" if false_positive_rate < 10 else "⚠️"
        
        print(f"{status} 误判率: {correct_results['false_positive']}/{correct_results['total']} = {false_positive_rate:.1f}% (目标: <10%)")
        print(f"   平均分数: {avg_score:.2f}")
    
    # 总体评估
    print("\n【总体评估】")
    
    type_a_pass = false_results['entity_replacement']['detected'] / false_results['entity_replacement']['total'] >= 0.8
    type_b_pass = false_results['causal_error']['detected'] / false_results['causal_error']['total'] >= 0.7
    type_c_pass = false_results['cross_document']['detected'] / false_results['cross_document']['total'] >= 0.6
    false_positive_pass = correct_results['false_positive'] / correct_results['total'] < 0.1
    
    all_pass = type_a_pass and type_b_pass and type_c_pass and false_positive_pass
    
    if all_pass:
        print("🎉 LLM Judge 模型通过所有测试！可以用于自动化评估。")
    else:
        print("⚠️  LLM Judge 模型部分测试未达标，建议：")
        if not type_a_pass:
            print("   - Type A 实体替换识别率不足，可能需要调整 prompt")
        if not type_b_pass:
            print("   - Type B 因果错误识别率不足，可能需要更强的模型")
        if not type_c_pass:
            print("   - Type C 跨片段拼接识别率不足（这是最难的，可以接受）")
        if not false_positive_pass:
            print("   - 误判率过高，Judge 过于严格")


def main():
    """主测试流程（异步版本）"""
    print("🚀 开始测试 LLM Judge 模型能力（异步批量版本）")
    print("=" * 80)
    
    # 1. 加载测试数据
    print("\n📦 加载测试数据...")
    false_claims = load_false_claims()
    correct_queries = load_correct_answers()
    print(f"✅ 加载 {len(false_claims)} 条错误答案")
    print(f"✅ 加载 {len(correct_queries)} 条正确答案")
    
    # 2. 异步运行测试
    asyncio.run(run_tests_async(false_claims, correct_queries))


async def run_tests_async(false_claims: List[Dict], correct_queries: List[Dict]):
    """异步运行所有测试"""
    # 初始化 AsyncOpenAI 客户端
    api_key = os.getenv("ALI_API_KEY")
    base_url = os.getenv("ALI_BASE_URL")
    
    if not api_key or not base_url:
        print("❌ 请设置环境变量 ALI_API_KEY 和 ALI_BASE_URL")
        return
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # 测试连接
    print("\n📡 测试 LLM Judge LLM 连接...")
    try:
        test_response = await client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": "请回复'已连接'"}],
            temperature=0.0
        )
        print(f"✅ Judge LLM 连接成功: {test_response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Judge LLM 连接失败: {e}")
        await client.close()
        return
    
    # 测试错误答案识别
    batch_size = 10
    false_results = await test_false_claims_async(client, false_claims, batch_size=batch_size)
    
    # 测试正确答案误判
    correct_results = await test_correct_answers_async(client, correct_queries, batch_size=batch_size)
    
    # 关闭客户端
    await client.close()
    
    # 打印总结
    print_summary(false_results, correct_results)
    
    # 保存结果
    output = {
        'meta': {
            'total_false_claims': len(false_claims),
            'total_correct_queries': len(correct_queries),
            'batch_size': batch_size,
            'timestamp': asyncio.get_event_loop().time()
        },
        'false_claims_results': {
            'entity_replacement': {
                'total': false_results['entity_replacement']['total'],
                'detected': false_results['entity_replacement']['detected'],
                'accuracy': false_results['entity_replacement']['detected'] / false_results['entity_replacement']['total'] if false_results['entity_replacement']['total'] > 0 else 0,
                'avg_score': sum(false_results['entity_replacement']['scores']) / len(false_results['entity_replacement']['scores']) if false_results['entity_replacement']['scores'] else 0,
                'scores': false_results['entity_replacement']['scores'],
                'details': false_results['entity_replacement']['details']
            },
            'causal_error': {
                'total': false_results['causal_error']['total'],
                'detected': false_results['causal_error']['detected'],
                'accuracy': false_results['causal_error']['detected'] / false_results['causal_error']['total'] if false_results['causal_error']['total'] > 0 else 0,
                'avg_score': sum(false_results['causal_error']['scores']) / len(false_results['causal_error']['scores']) if false_results['causal_error']['scores'] else 0,
                'scores': false_results['causal_error']['scores'],
                'details': false_results['causal_error']['details']
            },
            'cross_document': {
                'total': false_results['cross_document']['total'],
                'detected': false_results['cross_document']['detected'],
                'accuracy': false_results['cross_document']['detected'] / false_results['cross_document']['total'] if false_results['cross_document']['total'] > 0 else 0,
                'avg_score': sum(false_results['cross_document']['scores']) / len(false_results['cross_document']['scores']) if false_results['cross_document']['scores'] else 0,
                'scores': false_results['cross_document']['scores'],
                'details': false_results['cross_document']['details']
            }
        },
        'correct_answers_results': {
            'total': correct_results['total'],
            'false_positive': correct_results['false_positive'],
            'false_positive_rate': correct_results['false_positive'] / correct_results['total'] if correct_results['total'] > 0 else 0,
            'avg_score': sum(correct_results['scores']) / len(correct_results['scores']) if correct_results['scores'] else 0,
            'scores': correct_results['scores'],
            'details': correct_results['details']
        },
        'summary': {
            'entity_replacement_accuracy': false_results['entity_replacement']['detected'] / false_results['entity_replacement']['total'] if false_results['entity_replacement']['total'] > 0 else 0,
            'causal_error_accuracy': false_results['causal_error']['detected'] / false_results['causal_error']['total'] if false_results['causal_error']['total'] > 0 else 0,
            'cross_document_accuracy': false_results['cross_document']['detected'] / false_results['cross_document']['total'] if false_results['cross_document']['total'] > 0 else 0,
            'false_positive_rate': correct_results['false_positive'] / correct_results['total'] if correct_results['total'] > 0 else 0,
        }
    }
    
    output_file = 'llm_judge_test_results_detailed.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细测试结果已保存到 {output_file}")
    print(f"   - 包含所有 {len(false_claims)} 个错误 claims 的完整审核记录")
    print(f"   - 包含所有 {len(correct_queries)} 个正确答案的审核记录")
    print(f"   - 每条记录包含：prompt, judge_response, score, 判断结果")


if __name__ == "__main__":
    main()
