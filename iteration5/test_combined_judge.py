#!/usr/bin/env python3
"""
测试组合评估功能：一次 API 调用同时评估 Faithfulness 和 Relevance
"""
import asyncio
from evaluation import llm_combined_check, llm_combined_check_async, get_judge_llm


def test_sync_combined():
    """测试同步版本的组合评估"""
    print("="*80)
    print("测试同步组合评估（Faithfulness + Relevance）")
    print("="*80)
    
    llm = get_judge_llm()
    
    # 测试案例
    query = "SmartLock-100 的电池续航时间是多久？"
    
    # 案例 1: 完美答案（忠实 + 相关）
    answer_good = "SmartLock-100 使用 4 节 5 号电池，续航约 1 年。"
    chunks_good = [
        {"doc_id": "doc-1", "text": "SmartLock-100 使用 4 节 5 号电池，续航约 1 年。"},
        {"doc_id": "doc-1", "text": "SmartLock-100 支持指纹、密码、卡片、钥匙四种开锁方式。"}
    ]
    
    print(f"\n{'='*80}")
    print(f"测试案例 1: 完美答案")
    print(f"  问题: {query}")
    print(f"  答案: {answer_good}")
    print(f"  预期: Faithfulness=1.0, Relevance=1.0")
    print(f"{'='*80}\n")
    
    result = llm_combined_check(query, answer_good, chunks_good, llm)
    print(result['raw_response'])
    
    # 案例 2: 忠实但不相关
    answer_irrelevant = "SmartLock-100 支持指纹、密码、卡片、钥匙四种开锁方式。"
    
    print(f"\n\n{'='*80}")
    print(f"测试案例 2: 忠实但不相关")
    print(f"  问题: {query}")
    print(f"  答案: {answer_irrelevant}")
    print(f"  预期: Faithfulness=1.0, Relevance=0.0")
    print(f"{'='*80}\n")
    
    result = llm_combined_check(query, answer_irrelevant, chunks_good, llm)
    print(result['raw_response'])
    
    # 案例 3: 相关但不忠实（编造）
    answer_unfaithful = "SmartLock-100 使用 4 节 7 号电池，续航约 2 年。"
    
    print(f"\n\n{'='*80}")
    print(f"测试案例 3: 相关但不忠实")
    print(f"  问题: {query}")
    print(f"  答案: {answer_unfaithful}")
    print(f"  预期: Faithfulness=0.0, Relevance=1.0")
    print(f"{'='*80}\n")
    
    result = llm_combined_check(query, answer_unfaithful, chunks_good, llm)
    print(result['raw_response'])


async def test_async_combined():
    """测试异步版本的组合评估"""
    print("\n\n" + "="*80)
    print("测试异步批量组合评估")
    print("="*80)
    
    test_cases = [
        {
            "query": "SmartLock-100 的电池续航时间是多久？",
            "answer": "SmartLock-100 使用 4 节 5 号电池，续航约 1 年。",
            "chunks": [
                {"doc_id": "doc-1", "text": "SmartLock-100 使用 4 节 5 号电池，续航约 1 年。"}
            ],
            "expected": "Faithfulness=1.0, Relevance=1.0"
        },
        {
            "query": "SmartLock-100 的电池续航时间是多久？",
            "answer": "SmartLock-100 支持指纹、密码、卡片、钥匙四种开锁方式。",
            "chunks": [
                {"doc_id": "doc-1", "text": "SmartLock-100 支持指纹、密码、卡片、钥匙四种开锁方式。"}
            ],
            "expected": "Faithfulness=1.0, Relevance=0.0"
        },
        {
            "query": "SW-600 的电池型号是什么？",
            "answer": "未找到充分依据",
            "chunks": [
                {"doc_id": "doc-7", "text": "场景面板 SW-600 为四键无线场景开关，电池供电（CR2032×2）。"}
            ],
            "expected": "Faithfulness=低, Relevance=低"
        }
    ]
    
    # 创建并发任务
    tasks = []
    for case in test_cases:
        tasks.append(llm_combined_check_async(
            case['query'],
            case['answer'],
            case['chunks']
        ))
    
    # 并发执行
    print(f"\n🚀 并发评估 {len(tasks)} 个查询...")
    results = await asyncio.gather(*tasks)
    
    # 打印结果并提取分数
    for i, (case, result) in enumerate(zip(test_cases, results), 1):
        print(f"\n{'='*80}")
        print(f"查询 {i}")
        print(f"  问题: {case['query']}")
        print(f"  答案: {case['answer']}")
        print(f"  预期: {case['expected']}")
        print(f"{'='*80}")
        
        # 提取分数
        response_text = result['raw_response']
        import re
        
        faith_match = re.search(r'【Faithfulness 分数】\s*\n?\s*([0-9.]+)', response_text)
        rel_match = re.search(r'【Relevance 分数】\s*\n?\s*([0-9.]+)', response_text)
        
        if faith_match and rel_match:
            faith_score = float(faith_match.group(1))
            rel_score = float(rel_match.group(1))
            print(f"\n📊 评估结果:")
            print(f"   Faithfulness: {faith_score:.2f}")
            print(f"   Relevance:    {rel_score:.2f}")
        else:
            print(f"\n⚠️  无法提取分数")
        
        print(f"\n详细响应（前800字符）:")
        print(response_text[:800])
        print("...")


if __name__ == "__main__":
    print("🧪 开始测试组合评估功能\n")
    print("💰 成本优势: 一次 API 调用返回两个指标，节省 50% 成本\n")
    
    # 测试同步版本
    test_sync_combined()
    
    # 测试异步版本
    asyncio.run(test_async_combined())
    
    print("\n" + "="*80)
    print("✅ 所有测试完成！")
    print("="*80)
