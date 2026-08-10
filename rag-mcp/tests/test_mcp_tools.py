"""
测试 MCP Tools

测试 search_knowledge 和 generate_answer 两个接口。
"""

import os
import sys
import json
import asyncio

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_core import search_knowledge, generate_answer_with_retrieval


def load_rejection_config():
    """加载拒答配置"""
    config_path = os.path.join(os.path.dirname(__file__), '../rag_core/rejection_config.json')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 添加 Judge 模型配置
    config['judge_model'] = 'deepseek-chat'
    config['judge_base_url'] = 'https://api.deepseek.com'
    config['judge_api_key'] = os.getenv('DEEPSEEK_API_KEY')
    
    return config


def test_search_knowledge():
    """测试 search_knowledge - 纯检索"""
    print("="*60)
    print("测试 1: search_knowledge (纯检索)")
    print("="*60)
    
    rejection_config = load_rejection_config()
    
    # 测试查询
    queries = [
        "SmartLock-100 如何生成临时密码？",
        "这个产品防水吗？",  # 可能触发拒答
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        result = search_knowledge(
            query=query,
            top_k=5,
            retrieval_mode="hybrid",
            rerank=True,
            rejection_config=rejection_config
        )
        
        print(f"Total retrieved: {result['total_retrieved']}")
        print(f"Returned: {result['returned']}")
        print(f"Rejected: {result['rejected']}")
        
        if result['rejected']:
            print(f"Rejection reason: {result['rejection_reason']}")
        else:
            print(f"\nTop 3 results:")
            for i, r in enumerate(result['results'][:3], 1):
                print(f"\n{i}. Doc: {r['doc_id']}")
                print(f"   Score: {r['score']:.4f}")
                if 'rerank_score' in r['metadata']:
                    print(f"   Rerank: {r['metadata']['rerank_score']:.4f}")
                print(f"   Text: {r['text'][:80]}...")


async def test_generate_answer():
    """测试 generate_answer - 完整生成"""
    print("\n\n" + "="*60)
    print("测试 2: generate_answer (完整生成)")
    print("="*60)
    
    rejection_config = load_rejection_config()
    
    # 测试查询
    queries = [
        "SmartLock-100 如何生成临时密码？",
        "SmartCam-200 的红外夜视距离多远？",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        
        result = await generate_answer_with_retrieval(
            query=query,
            top_k=5,
            retrieval_mode="hybrid",
            rerank=True,
            rejection_config=rejection_config
        )
        
        print(f"Retrieved: {result['retrieved_count']}")
        print(f"Rejected: {result['rejected']}")
        
        if result['rejected']:
            print(f"Rejection reason: {result['rejection_reason']}")
        
        print(f"\nAnswer: {result['answer']}")
        
        if result['citations']:
            print(f"\nCitations ({len(result['citations'])}):")
            for i, cit in enumerate(result['citations'], 1):
                print(f"  {i}. [{cit['span']}] from {cit['source']}")
        
        if result['faithfulness_score']:
            print(f"\nFaithfulness: {result['faithfulness_score']:.3f}")
        if result['relevance_score']:
            print(f"Relevance: {result['relevance_score']:.3f}")


def test_mcp_tool_schema():
    """测试 MCP Tool 的 Schema 定义"""
    print("\n\n" + "="*60)
    print("测试 3: MCP Tool Schema")
    print("="*60)
    
    # search_knowledge schema
    search_schema = {
        "name": "search_knowledge",
        "description": "纯检索知识库，不做生成",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 10, "maximum": 20}
            },
            "required": ["query"]
        }
    }
    
    print("\n1. search_knowledge Schema:")
    print(json.dumps(search_schema, indent=2, ensure_ascii=False))
    
    # generate_answer schema
    generate_schema = {
        "name": "generate_answer",
        "description": "完整生成：检索 + 生成 + 校验",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5, "maximum": 10}
            },
            "required": ["query"]
        }
    }
    
    print("\n2. generate_answer Schema:")
    print(json.dumps(generate_schema, indent=2, ensure_ascii=False))


def main():
    """运行所有测试"""
    print("开始测试 MCP Tools...")
    
    # 测试 1: 检索
    test_search_knowledge()
    
    # 测试 2: 生成（异步）
    asyncio.run(test_generate_answer())
    
    # 测试 3: Schema
    test_mcp_tool_schema()
    
    print("\n\n" + "="*60)
    print("所有测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
