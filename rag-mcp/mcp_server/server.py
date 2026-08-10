"""
MCP Server for RAG System

暴露两个 MCP Tools：
1. search_knowledge - 纯检索，不生成
2. generate_answer - 完整生成（检索 + 生成 + 校验）

这是一个薄薄的协议层，主要逻辑在 rag_core 中。
"""

import os
import sys
import json
import asyncio
from typing import Any, Dict

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

from rag_core import search_knowledge, generate_answer_with_retrieval


# 创建 MCP Server 实例
app = Server("rag-server")

# 全局配置（从环境变量或配置文件加载）
RAG_CONFIG = {
    "corpus_dir": os.getenv("RAG_CORPUS_DIR", None),  # 如果 None，使用默认路径
    "chunking_strategy": os.getenv("RAG_CHUNKING_STRATEGY", "fixed_100_50"),
    "retrieval_mode": os.getenv("RAG_RETRIEVAL_MODE", "hybrid"),
    "rerank": os.getenv("RAG_RERANK", "true").lower() == "true",
}

# 加载拒答配置
def load_rejection_config() -> Dict:
    """加载拒答配置"""
    config_path = os.path.join(os.path.dirname(__file__), '../rag_core/rejection_config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        # 默认配置
        config = {
            "rejection_enabled": True,
            "rejection_layers": {
                "layer1_rerank": {
                    "enabled": True,
                    "top1_threshold": 0.50,
                    "top3_avg_threshold": 0.45
                },
                "layer3_judge": {
                    "enabled": True,
                    "faithfulness_threshold": 0.80,
                    "relevance_threshold": 0.75
                }
            }
        }
    
    # 添加 Judge 模型配置（从环境变量读取）
    config['judge_model'] = os.getenv('RAG_JUDGE_MODEL', 'deepseek-chat')
    config['judge_base_url'] = os.getenv('RAG_JUDGE_BASE_URL', 'https://api.deepseek.com')
    config['judge_api_key'] = os.getenv('DEEPSEEK_API_KEY')
    
    return config


REJECTION_CONFIG = load_rejection_config()


@app.list_tools()
async def list_tools() -> list[Tool]:
    """列出可用的 MCP Tools"""
    return [
        Tool(
            name="search_knowledge",
            description=(
                "纯检索知识库，不做生成。用于判断知识库里有没有相关内容、够不够回答。"
                "支持 Layer 0/1 拒答检查（基于 rerank 分数）。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户问题或搜索关键词"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回片段数，默认 10，最大 20",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="generate_answer",
            description=(
                "完整生成：检索 + 两段式生成 + 确定性校验。"
                "返回带引用标注的答案，支持完整的四层拒答机制（Layer 0-3）。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户问题"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "检索片段数，默认 5",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """处理 MCP Tool 调用"""
    
    if name == "search_knowledge":
        # 纯检索
        query = arguments.get("query")
        top_k = arguments.get("top_k", 10)
        
        if not query:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "query is required"}, ensure_ascii=False)
            )]
        
        # 调用 rag_core 的检索函数
        result = search_knowledge(
            query=query,
            top_k=top_k,
            retrieval_mode=RAG_CONFIG["retrieval_mode"],
            rerank=RAG_CONFIG["rerank"],
            corpus_dir=RAG_CONFIG["corpus_dir"],
            chunking_strategy=RAG_CONFIG["chunking_strategy"],
            rejection_config=REJECTION_CONFIG
        )
        
        # 返回 JSON 格式结果
        return [TextContent(
            type="text",
            text=json.dumps(result, ensure_ascii=False, indent=2)
        )]
    
    elif name == "generate_answer":
        # 完整生成
        query = arguments.get("query")
        top_k = arguments.get("top_k", 5)
        
        if not query:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "query is required"}, ensure_ascii=False)
            )]
        
        # 调用 rag_core 的生成函数（异步）
        result = await generate_answer_with_retrieval(
            query=query,
            top_k=top_k,
            retrieval_mode=RAG_CONFIG["retrieval_mode"],
            rerank=RAG_CONFIG["rerank"],
            corpus_dir=RAG_CONFIG["corpus_dir"],
            chunking_strategy=RAG_CONFIG["chunking_strategy"],
            rejection_config=REJECTION_CONFIG
        )
        
        # 返回 JSON 格式结果
        return [TextContent(
            type="text",
            text=json.dumps(result, ensure_ascii=False, indent=2)
        )]
    
    else:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)
        )]


async def main():
    """启动 MCP Server"""
    # 使用 stdio 传输（标准输入/输出）
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    # 运行 MCP Server
    asyncio.run(main())
