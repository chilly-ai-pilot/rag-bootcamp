#!/bin/bash
# RAG MCP Server 启动脚本

set -e

echo "=========================================="
echo "  RAG MCP Server"
echo "=========================================="
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python not found"
    exit 1
fi

# 检查环境变量
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "❌ Error: DEEPSEEK_API_KEY not set"
    echo ""
    echo "Please set it:"
    echo "  export DEEPSEEK_API_KEY='your-api-key'"
    exit 1
fi

echo "✓ Python: $(python3 --version)"
echo "✓ API Key: configured"
echo ""

# 设置默认值
export RAG_CORPUS_DIR="${RAG_CORPUS_DIR:-$(dirname $0)/../iteration8/corpus}"
export RAG_CHUNKING_STRATEGY="${RAG_CHUNKING_STRATEGY:-fixed_100_50}"
export RAG_RETRIEVAL_MODE="${RAG_RETRIEVAL_MODE:-hybrid}"
export RAG_RERANK="${RAG_RERANK:-true}"

echo "Configuration:"
echo "  Corpus: $RAG_CORPUS_DIR"
echo "  Chunking: $RAG_CHUNKING_STRATEGY"
echo "  Retrieval: $RAG_RETRIEVAL_MODE"
echo "  Rerank: $RAG_RERANK"
echo ""
echo "=========================================="
echo "Starting MCP Server..."
echo "=========================================="
echo ""

# 启动服务
cd "$(dirname $0)"
exec python3 mcp_server/server.py
