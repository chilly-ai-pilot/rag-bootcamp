#!/usr/bin/env python3
"""
快速测试脚本

在不启动 MCP Server 的情况下，直接测试 search_knowledge 和 generate_answer。
"""

import os
import sys
import json
import asyncio

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from rag_core import search_knowledge, generate_answer_with_retrieval


def load_config():
    """加载配置"""
    config_path = os.path.join(os.path.dirname(__file__), 'rag_core/rejection_config.json')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 添加 Judge 配置
    config['judge_model'] = 'deepseek-chat'
    config['judge_base_url'] = 'https://api.deepseek.com'
    config['judge_api_key'] = os.getenv('DEEPSEEK_API_KEY')
    
    return config


async def quick_test():
    """快速测试两个接口"""
    
    print("="*70)
    print("RAG MCP Server - 快速测试")
    print("="*70)
    
    config = load_config()
    
    # 测试查询
    query = "SmartLock-100 如何生成临时密码？"
    
    # ==================== 测试 1: 纯检索 ====================
    print(f"\n{'='*70}")
    print("测试 1: search_knowledge (纯检索)")
    print(f"{'='*70}")
    print(f"Query: {query}\n")
    
    search_result = search_knowledge(
        query=query,
        top_k=5,
        retrieval_mode="hybrid",
        rerank=True,
        rejection_config=config
    )
    
    print(f"✓ Total retrieved: {search_result['total_retrieved']}")
    print(f"✓ Returned: {search_result['returned']}")
    print(f"✓ Rejected: {search_result['rejected']}")
    
    if search_result['rejected']:
        print(f"✗ Rejection reason: {search_result['rejection_reason']}")
    else:
        print(f"\n📄 Top 3 results:")
        for i, r in enumerate(search_result['results'][:3], 1):
            print(f"\n{i}. {r['doc_id']} (score: {r['score']:.4f})")
            if 'rerank_score' in r['metadata']:
                print(f"   Rerank: {r['metadata']['rerank_score']:.4f}")
            print(f"   {r['text'][:100]}...")
    
    # ==================== 测试 2: 完整生成 ====================
    print(f"\n\n{'='*70}")
    print("测试 2: generate_answer (完整生成)")
    print(f"{'='*70}")
    print(f"Query: {query}\n")
    
    gen_result = await generate_answer_with_retrieval(
        query=query,
        top_k=5,
        retrieval_mode="hybrid",
        rerank=True,
        rejection_config=config
    )
    
    print(f"✓ Retrieved: {gen_result['retrieved_count']}")
    print(f"✓ Rejected: {gen_result['rejected']}")
    
    if gen_result['rejected']:
        print(f"✗ Rejection reason: {gen_result['rejection_reason']}")
    
    print(f"\n💬 Answer:")
    print(f"   {gen_result['answer']}")
    
    if gen_result['citations']:
        print(f"\n📌 Citations ({len(gen_result['citations'])}):")
        for i, cit in enumerate(gen_result['citations'], 1):
            print(f"   {i}. [{cit['span']}]")
            print(f"      Source: {cit['source']}")
    
    if gen_result['faithfulness_score']:
        print(f"\n📊 Metrics:")
        print(f"   Faithfulness: {gen_result['faithfulness_score']:.3f}")
        if gen_result['relevance_score']:
            print(f"   Relevance: {gen_result['relevance_score']:.3f}")
    
    print(f"\n{'='*70}")
    print("✅ 测试完成！")
    print(f"{'='*70}\n")


def main():
    """主函数"""
    # 检查 API Key
    if not os.getenv('DEEPSEEK_API_KEY'):
        print("❌ 错误: 请设置环境变量 DEEPSEEK_API_KEY")
        print("   export DEEPSEEK_API_KEY='your-api-key'")
        sys.exit(1)
    
    # 运行测试
    asyncio.run(quick_test())


if __name__ == "__main__":
    main()
