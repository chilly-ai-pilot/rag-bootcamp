"""
HTTP/SSE Server for RAG MCP

支持两种传输方式：
1. stdio - MCP 标准协议（用于 Kiro 等客户端）
2. HTTP/SSE - RESTful API + Server-Sent Events（用于 Web 集成）

HTTP 端点：
- POST /search - 搜索知识库
- POST /generate - 生成答案（支持 streaming）
- POST /generate/stream - 流式生成答案（SSE）
"""

import os
import sys
import json
import asyncio
from typing import AsyncIterator
from contextlib import asynccontextmanager

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_core import search_knowledge, generate_answer_with_retrieval


# 全局配置
RAG_CONFIG = {
    "corpus_dir": os.getenv("RAG_CORPUS_DIR", None),
    "chunking_strategy": os.getenv("RAG_CHUNKING_STRATEGY", "fixed_100_50"),
    "retrieval_mode": os.getenv("RAG_RETRIEVAL_MODE", "hybrid"),
    "rerank": os.getenv("RAG_RERANK", "true").lower() == "true",
}


# 加载拒答配置
def load_rejection_config():
    """加载拒答配置"""
    config_path = os.path.join(os.path.dirname(__file__), '../rag_core/rejection_config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
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
    
    config['judge_model'] = os.getenv('RAG_JUDGE_MODEL', 'deepseek-chat')
    config['judge_base_url'] = os.getenv('RAG_JUDGE_BASE_URL', 'https://api.deepseek.com')
    config['judge_api_key'] = os.getenv('DEEPSEEK_API_KEY')
    
    return config


REJECTION_CONFIG = load_rejection_config()


# Pydantic 模型
class SearchRequest(BaseModel):
    query: str = Field(..., description="用户问题或搜索关键词")
    top_k: int = Field(10, ge=1, le=20, description="返回片段数")


class GenerateRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    top_k: int = Field(5, ge=1, le=10, description="检索片段数")
    stream: bool = Field(False, description="是否使用流式输出")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("🚀 RAG HTTP Server 启动")
    print(f"   Corpus: {RAG_CONFIG['corpus_dir'] or 'default'}")
    print(f"   Chunking: {RAG_CONFIG['chunking_strategy']}")
    print(f"   Retrieval: {RAG_CONFIG['retrieval_mode']}")
    print(f"   Rerank: {RAG_CONFIG['rerank']}")
    yield
    print("👋 RAG HTTP Server 关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="RAG MCP HTTP Server",
    description="RAG System HTTP/SSE API",
    version="1.0.0",
    lifespan=lifespan
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """健康检查"""
    return {
        "service": "RAG MCP HTTP Server",
        "version": "1.0.0",
        "status": "running",
        "config": RAG_CONFIG
    }


@app.post("/search")
async def search(request: SearchRequest):
    """搜索知识库（纯检索）"""
    try:
        result = search_knowledge(
            query=request.query,
            top_k=request.top_k,
            retrieval_mode=RAG_CONFIG["retrieval_mode"],
            rerank=RAG_CONFIG["rerank"],
            corpus_dir=RAG_CONFIG["corpus_dir"],
            chunking_strategy=RAG_CONFIG["chunking_strategy"],
            rejection_config=REJECTION_CONFIG
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate(request: GenerateRequest):
    """生成答案（完整流程）"""
    try:
        result = await generate_answer_with_retrieval(
            query=request.query,
            top_k=request.top_k,
            retrieval_mode=RAG_CONFIG["retrieval_mode"],
            rerank=RAG_CONFIG["rerank"],
            corpus_dir=RAG_CONFIG["corpus_dir"],
            chunking_strategy=RAG_CONFIG["chunking_strategy"],
            rejection_config=REJECTION_CONFIG
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_stream(query: str, top_k: int) -> AsyncIterator[str]:
    """流式生成答案
    
    使用 Server-Sent Events (SSE) 格式
    """
    try:
        # 1. 先发送检索状态
        yield f"data: {json.dumps({'type': 'status', 'message': '正在检索...', 'stage': 'retrieval'}, ensure_ascii=False)}\n\n"
        
        # 2. 生成答案（这里可以修改 generate_answer_with_retrieval 支持流式）
        result = await generate_answer_with_retrieval(
            query=query,
            top_k=top_k,
            retrieval_mode=RAG_CONFIG["retrieval_mode"],
            rerank=RAG_CONFIG["rerank"],
            corpus_dir=RAG_CONFIG["corpus_dir"],
            chunking_strategy=RAG_CONFIG["chunking_strategy"],
            rejection_config=REJECTION_CONFIG
        )
        
        # 3. 发送检索结果
        if result.get('rejected'):
            yield f"data: {json.dumps({'type': 'rejection', 'reason': result.get('rejection_reason')}, ensure_ascii=False)}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'status', 'message': '正在生成答案...', 'stage': 'generation'}, ensure_ascii=False)}\n\n"
            
            # 4. 逐字发送答案（模拟流式输出）
            answer = result.get('answer', '')
            for i, char in enumerate(answer):
                yield f"data: {json.dumps({'type': 'token', 'content': char, 'position': i}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)  # 模拟延迟
            
            # 5. 发送引用信息
            if result.get('citations'):
                yield f"data: {json.dumps({'type': 'citations', 'data': result['citations']}, ensure_ascii=False)}\n\n"
        
        # 6. 发送完成信号
        yield f"data: {json.dumps({'type': 'done', 'result': result}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"


@app.post("/generate/stream")
async def generate_streaming(request: GenerateRequest):
    """流式生成答案（SSE）"""
    return StreamingResponse(
        generate_stream(request.query, request.top_k),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用 nginx 缓冲
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # 从环境变量读取配置
    host = os.getenv("RAG_HTTP_HOST", "0.0.0.0")
    port = int(os.getenv("RAG_HTTP_PORT", "8000"))
    
    print(f"Starting RAG HTTP Server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
