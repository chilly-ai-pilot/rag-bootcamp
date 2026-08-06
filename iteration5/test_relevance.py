#!/usr/bin/env python3
"""
快速测试 Answer Relevance 功能
"""
import asyncio
from evaluation import llm_relevance_check, llm_relevance_check_async, get_judge_llm


def test_sync_relevance():
    """测试同步版本的 relevance 检查"""
    print("="*80)
    print("测试同步版本的 Answer Relevance 检查")
    print("="*80)
    
    llm = get_judge_llm()
    
    test_cases = [
        {
            "query": "SmartLock-100 的电池续航时间是多久？",
            "answer": "SmartLock-100 使用 4 节 5 号电池，续航约 1 年。",
            "expected": "高相关性（完整回答了问题）"
        },
        {
            "query": "SmartLock-100 的电池续航时间是多久？",
            "answer": "SmartLock-100 支持指纹、密码、卡片、钥匙四种开锁方式。",
            "expected": "不相关（答非所问）"
        },
        {
            "query": "SW-600 的电池型号是什么？",
            "answer": "未找到充分依据",
            "expected": "低相关性（未回答问题）"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"测试案例 {i}:")
        print(f"  问题: {case['query']}")
        print(f"  答案: {case['answer']}")
        print(f"  预期: {case['expected']}")
        print(f"{'='*80}\n")
        
        result = llm_relevance_check(case['query'], case['answer'], llm)
        print(result['raw_response'])
        print()


async def test_async_relevance():
    """测试异步版本的 relevance 检查"""
    print("="*80)
    print("测试异步批量 Answer Relevance 检查")
    print("="*80)
    
    test_cases = [
        {
            "query": "SmartLock-100 的电池续航时间是多久？",
            "answer": "SmartLock-100 使用 4 节 5 号电池，续航约 1 年。"
        },
        {
            "query": "SmartLock-100 如何开锁？",
            "answer": "SmartLock-100 支持指纹、密码、卡片、钥匙四种开锁方式。"
        },
        {
            "query": "SW-600 的电池型号是什么？",
            "answer": "未找到充分依据"
        }
    ]
    
    # 创建并发任务
    tasks = []
    for case in test_cases:
        tasks.append(llm_relevance_check_async(case['query'], case['answer']))
    
    # 并发执行
    print(f"\n🚀 并发评估 {len(tasks)} 个查询...")
    results = await asyncio.gather(*tasks)
    
    # 打印结果
    for i, (case, result) in enumerate(zip(test_cases, results), 1):
        print(f"\n{'='*80}")
        print(f"查询 {i}: {case['query']}")
        print(f"答案: {case['answer']}")
        print(f"{'='*80}")
        
        # 提取分数
        response_text = result['raw_response']
        import re
        match = re.search(r'【Answer Relevance 分数】\s*\n?\s*([0-9.]+)', response_text)
        if match:
            score = float(match.group(1))
            print(f"\n📊 相关性分数: {score:.2f}")
        
        print(f"\n{response_text[:500]}...\n")


if __name__ == "__main__":
    print("🧪 开始测试 Answer Relevance 功能\n")
    
    # 测试同步版本
    test_sync_relevance()
    
    # 测试异步版本
    print("\n" + "="*80 + "\n")
    asyncio.run(test_async_relevance())
    
    print("\n✅ 所有测试完成！")
