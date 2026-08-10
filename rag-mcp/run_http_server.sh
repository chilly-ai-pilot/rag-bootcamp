#!/bin/bash

# RAG MCP HTTP Server 启动脚本

# 设置默认配置
export RAG_HTTP_HOST="${RAG_HTTP_HOST:-0.0.0.0}"
export RAG_HTTP_PORT="${RAG_HTTP_PORT:-8000}"
export RAG_CORPUS_DIR="${RAG_CORPUS_DIR:-../iteration8/corpus}"
export RAG_CHUNKING_STRATEGY="${RAG_CHUNKING_STRATEGY:-fixed_100_50}"
export RAG_RETRIEVAL_MODE="${RAG_RETRIEVAL_MODE:-hybrid}"
export RAG_RERANK="${RAG_RERANK:-true}"

# 检查 API Key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "⚠️  警告: DEEPSEEK_API_KEY 未设置"
    echo "   某些功能（如 Judge 评估）可能无法使用"
fi

echo "🚀 启动 RAG HTTP Server"
echo "   Host: $RAG_HTTP_HOST"
echo "   Port: $RAG_HTTP_PORT"
echo "   Corpus: $RAG_CORPUS_DIR"
echo ""

# 启动服务器
cd "$(dirname "$0")"
python3 mcp_server/http_server.py
