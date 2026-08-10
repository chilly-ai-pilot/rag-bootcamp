# RAG MCP Server

将 RAG 系统功能暴露为 MCP (Model Context Protocol) Tools，支持 Agent 灵活编排检索和生成流程。

## 🎯 设计理念

### 为什么拆成两个 Tool？

Agent 需要灵活编排，拆开后能先检索判断信息够不够，不够再上网搜，搜完再生成。一个 Tool 做不到这种按需组合。

同时和四层拒答体系对齐：
- **检索 Tool** (`search_knowledge`) 触发 **Layer 0/1** 拒答
- **生成 Tool** (`generate_answer`) 触发 **Layer 2/3** 拒答

Agent 在检索阶段就能提前走降级策略。

## 📁 目录结构

```
rag-mcp/
├── rag_core/                  # RAG 核心逻辑
│   ├── search.py              # 纯检索函数
│   ├── generate.py            # 完整生成函数
│   ├── rejection_config.json  # 拒答配置
│   └── __init__.py
│
├── mcp_server/                # MCP 协议层（薄薄一层）
│   ├── server.py              # MCP Server 入口
│   └── __init__.py
│
├── tests/                     # 测试
│   └── test_mcp_tools.py      # 测试 MCP Tool 调用
│
├── requirements.txt
└── README.md
```

**依赖 iteration8**：rag_core 中的函数依赖 iteration8 的核心模块（chunking, retrieval, generation, scoring, evaluation）。

## 🔧 安装

```bash
# 1. 安装依赖
cd rag-mcp
pip install -r requirements.txt

# 2. 设置环境变量
export DEEPSEEK_API_KEY="your-api-key"

# 可选：自定义配置
export RAG_CORPUS_DIR="/path/to/corpus"
export RAG_CHUNKING_STRATEGY="fixed_100_50"
export RAG_RETRIEVAL_MODE="hybrid"
export RAG_RERANK="true"
```

## 🚀 使用方式

### 方式 1: 测试 MCP Tools（不启动 Server）

```bash
cd rag-mcp
python tests/test_mcp_tools.py
```

这会直接调用 `rag_core` 的函数，测试检索和生成功能。

### 方式 2: 启动 MCP Server

```bash
cd rag-mcp
python mcp_server/server.py
```

Server 启动后，通过 stdio 接收 MCP 协议消息。

### 方式 3: 在 Kiro 中配置

在 Kiro 的 `mcp.json` 中添加：

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["/path/to/rag-mcp/mcp_server/server.py"],
      "env": {
        "DEEPSEEK_API_KEY": "your-api-key",
        "RAG_CORPUS_DIR": "/path/to/corpus"
      }
    }
  }
}
```

## 📡 MCP Tools 接口

### 1. `search_knowledge` — 纯检索

只检索知识库，不做生成。Agent 用来判断"知识库里有没有相关内容、够不够回答"。

**输入：**
```json
{
  "query": "用户问题或搜索关键词",
  "top_k": 10  // 返回片段数，默认 10，最大 20
}
```

**输出：**
```json
{
  "results": [
    {
      "chunk_id": "doc1_chunk_0",
      "text": "文档片段内容...",
      "score": 0.85,
      "doc_id": "doc1",
      "metadata": {
        "start": 0,
        "end": 200,
        "rerank_score": 0.92
      }
    }
  ],
  "total_retrieved": 40,     // 粗筛候选集大小
  "returned": 10,            // 实际返回数量
  "rejected": false,         // 是否触发检索层拒答
  "rejection_reason": null,  // 拒答原因（layer0/layer1）
  "query": "用户问题"
}
```

**拒答机制：**
- **Layer 1**: 基于 rerank 分数判断
  - Top-1 分数过低
  - Top-3 平均分数过低

### 2. `generate_answer` — 完整生成

检索 + 两段式生成 + 确定性校验。Agent 用来获取最终答案。

**输入：**
```json
{
  "query": "用户问题",
  "top_k": 5  // 检索片段数，默认 5
}
```

**输出：**
```json
{
  "answer": "带引用标注的完整答案【1】【2】",
  "citations": [
    {
      "span": "被引用的文字",
      "source": "doc1:0-200",
      "chunk_id": "doc1_chunk_0"
    }
  ],
  "rejected": false,          // 是否触发拒答
  "rejection_reason": null,   // 拒答原因（layer0-3）
  "faithfulness_score": 0.92, // Judge 模型打分
  "relevance_score": 0.88,
  "query": "用户问题",
  "retrieved_count": 5,       // 检索到的文档数
  "retrieval_rejected": false // 检索阶段是否被拒
}
```

**拒答机制（四层）：**
- **Layer 0**: 检索命中检查（需要 ground truth，仅评估模式）
- **Layer 1**: Rerank 分数检查（检索阶段）
- **Layer 2**: Citation 确定性校验（生成阶段）
- **Layer 3**: Judge 模型评分（Faithfulness & Relevance）

## 🎭 使用场景

### 场景 1: Agent 先检索判断

```python
# Agent 先调用检索判断信息是否充足
search_result = search_knowledge(query="产品防水吗？", top_k=5)

if search_result["rejected"] or search_result["returned"] == 0:
    # 知识库信息不足，走降级策略
    # 1. 上网搜索
    # 2. 或直接告诉用户无法回答
    web_search(query)
else:
    # 信息充足，生成答案
    answer = generate_answer(query="产品防水吗？", top_k=5)
```

### 场景 2: 混合检索（RAG + Web）

```python
# 1. 先从知识库检索
kb_results = search_knowledge(query="最新产品功能", top_k=10)

# 2. 同时从网页检索
web_results = web_search(query="最新产品功能")

# 3. 合并结果后生成
combined_context = merge_results(kb_results, web_results)
answer = llm_generate(query, combined_context)
```

### 场景 3: 直接生成（信任知识库）

```python
# 如果确信知识库足够，直接生成
answer = generate_answer(query="产品规格参数", top_k=5)

if answer["rejected"]:
    # 被拒答，走降级
    fallback_response()
```

## ⚙️ 配置说明

### 拒答配置 (`rejection_config.json`)

```json
{
  "rejection_enabled": true,
  "rejection_layers": {
    "layer1_rerank": {
      "enabled": true,
      "top1_threshold": 0.50,      // Top-1 分数阈值
      "top3_avg_threshold": 0.45   // Top-3 平均分数阈值
    },
    "layer3_judge": {
      "enabled": true,
      "faithfulness_threshold": 0.80,
      "relevance_threshold": 0.75
    }
  },
  "rejection_message": "抱歉，我在提供的资料中未找到足够充分的信息..."
}
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API Key（必需） | - |
| `RAG_CORPUS_DIR` | 语料库目录 | `iteration8/corpus` |
| `RAG_CHUNKING_STRATEGY` | 分块策略 | `fixed_100_50` |
| `RAG_RETRIEVAL_MODE` | 检索模式 | `hybrid` |
| `RAG_RERANK` | 是否使用 rerank | `true` |
| `RAG_JUDGE_MODEL` | Judge 模型 | `deepseek-chat` |

## 🧪 测试

```bash
# 运行测试
cd rag-mcp
python tests/test_mcp_tools.py

# 测试会执行：
# 1. search_knowledge - 纯检索测试
# 2. generate_answer - 完整生成测试
# 3. MCP Tool Schema 验证
```

## 📊 性能特点

- **低延迟检索**: 纯检索 ~200ms（hybrid + rerank）
- **完整生成**: ~2-3s（检索 + 生成 + Judge 评估）
- **拒答率**: 根据配置，通常 5-15%
- **Faithfulness**: 0.85-0.95（Judge 评估）

## 🔄 与 iteration8 的关系

`rag-mcp` 是 `iteration8` 的 **MCP 封装层**：

- **rag_core**: 调用 iteration8 的核心模块
- **mcp_server**: 将功能暴露为 MCP Tools
- **tests**: 独立测试 MCP Tools

**不重复实现**核心逻辑，只是提供不同的接口形式。

## 📝 后续扩展

- [ ] 支持流式生成（streaming）
- [ ] 添加缓存层（减少重复检索）
- [ ] 支持多轮对话上下文
- [ ] 添加更多检索策略（如 ColBERT）
- [ ] 支持自定义 Judge 模型

## 📄 License

同 RAG Bootcamp 项目
