# RAG MCP Server 架构设计

## 📐 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Agent / LLM                            │
│                   (Kiro, Claude, etc.)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ MCP Protocol
                       │ (stdio / JSON-RPC)
┌──────────────────────▼──────────────────────────────────────┐
│                   MCP Server Layer                          │
│              (mcp_server/server.py)                         │
│  ┌─────────────────────────┬─────────────────────────────┐ │
│  │  search_knowledge       │   generate_answer           │ │
│  │  (Tool)                 │   (Tool)                    │ │
│  └────────┬────────────────┴──────────────┬──────────────┘ │
└───────────┼────────────────────────────────┼────────────────┘
            │                                │
            │ 调用 rag_core 函数             │
            │                                │
┌───────────▼────────────────────────────────▼────────────────┐
│                    RAG Core Layer                           │
│                   (rag_core/)                               │
│  ┌──────────────────────┐  ┌───────────────────────────┐   │
│  │  search.py           │  │  generate.py              │   │
│  │  - search_knowledge  │  │  - generate_answer        │   │
│  │  - Layer 0/1 拒答    │  │  - Layer 0-3 拒答         │   │
│  └──────────┬───────────┘  └──────────┬────────────────┘   │
│             │                          │                     │
│             └──────────┬───────────────┘                     │
│                        │                                     │
│          调用 iteration8 核心模块                            │
│                        │                                     │
└────────────────────────┼─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                  Iteration8 Core Modules                     │
│              (../iteration8/)                                │
│  ┌──────────────┬──────────────┬──────────────┬───────────┐ │
│  │ chunking.py  │ retrieval.py │ generation.py│ scoring.py│ │
│  │ - 文档分块   │ - 检索引擎   │ - 答案生成   │ - 评分    │ │
│  └──────────────┴──────────────┴──────────────┴───────────┘ │
│  ┌──────────────────────────────────────────────────────────┐│
│  │ evaluation.py - Judge 评估 (Faithfulness/Relevance)     ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

## 🔀 数据流

### Flow 1: search_knowledge (纯检索)

```
Agent Request
    ↓
MCP Server (search_knowledge tool)
    ↓
rag_core/search.py
    ↓
├─ chunking.py: 构建文档块
├─ retrieval.py: 检索候选集 (vector/bm25/hybrid)
├─ retrieval.py: Rerank (可选)
└─ Layer 0/1 拒答检查
    ↓
返回检索结果 (JSON)
    ↓
Agent 判断信息是否充足
```

### Flow 2: generate_answer (完整生成)

```
Agent Request
    ↓
MCP Server (generate_answer tool)
    ↓
rag_core/generate.py
    ↓
Step 1: 检索阶段
├─ chunking.py: 构建文档块
├─ retrieval.py: 检索候选集
├─ retrieval.py: Rerank
└─ Layer 1 拒答检查 (rerank 分数)
    ↓
    如果检索被拒 → 返回拒答消息
    ↓
Step 2: 生成阶段
├─ generation.py: 两段式生成
│   ├─ 第一段: 推理 + 引用选择
│   └─ 第二段: 格式化答案
├─ Layer 2: Citation 校验 (确定性)
└─ Layer 3: Judge 评估 (Faithfulness/Relevance)
    ↓
返回答案 + 引用 + 评分 (JSON)
    ↓
Agent 使用答案或走降级
```

## 🛡️ 四层拒答机制

### Layer 0: 检索命中检查
- **位置**: rag_core/generate.py (仅评估模式)
- **条件**: 需要 ground truth
- **判断**: 检索结果是否命中真实答案
- **触发**: hit=0 或 rank > threshold

### Layer 1: Rerank 分数检查
- **位置**: rag_core/search.py, rag_core/generate.py
- **条件**: 启用 rerank
- **判断**: 
  - Top-1 分数 < threshold (默认 0.50)
  - Top-3 平均分数 < threshold (默认 0.45)
- **触发**: 检索阶段拒答

### Layer 2: Citation 确定性校验
- **位置**: generation.py (generate_answer_async)
- **条件**: 生成答案中包含引用标注
- **判断**: 引用文字与检索文档的相似度
- **触发**: 引用不准确 → 拒答

### Layer 3: Judge 模型评估
- **位置**: generation.py (generate_answer_async)
- **条件**: Judge 模型启用
- **判断**:
  - Faithfulness < threshold (默认 0.80)
  - Relevance < threshold (默认 0.75)
- **触发**: 生成质量不合格 → 拒答

## 🔧 模块职责

### MCP Server Layer (`mcp_server/`)
**职责**: 薄薄的协议层，只负责 MCP 协议处理
- 接收 MCP 请求（JSON-RPC over stdio）
- 参数校验和转换
- 调用 rag_core 函数
- 返回 MCP 响应

**不做**:
- ❌ 不实现业务逻辑
- ❌ 不做检索或生成
- ❌ 不做拒答判断

### RAG Core Layer (`rag_core/`)
**职责**: RAG 业务逻辑，但不重复实现基础功能
- **search.py**: 组织检索流程 + Layer 0/1 拒答
- **generate.py**: 组织生成流程 + 调用 generation.py
- **配置管理**: rejection_config.json

**依赖**:
- ✅ 依赖 iteration8 的核心模块
- ✅ 不重复实现 chunking/retrieval/generation

### Iteration8 Core (`../iteration8/`)
**职责**: 基础 RAG 功能实现
- **chunking.py**: 文档分块
- **retrieval.py**: 检索引擎 (vector/bm25/hybrid/rerank)
- **generation.py**: 两段式生成 + Citation 校验
- **evaluation.py**: Judge 评估
- **scoring.py**: 评分计算

## 🎯 设计原则

### 1. 关注点分离
- **MCP Server**: 只管协议，不管业务
- **RAG Core**: 只管流程，不管实现
- **Iteration8**: 只管功能，不管接口

### 2. 不重复实现
- 复用 iteration8 的成熟代码
- rag_core 只做流程编排和拒答判断
- 新增功能在 iteration8 中实现

### 3. 灵活编排
- 拆分检索和生成两个独立 Tool
- Agent 可以自由组合调用
- 支持混合检索场景（RAG + Web）

### 4. 配置驱动
- 环境变量控制行为
- rejection_config.json 统一配置
- 方便 CI/CD 和多环境部署

## 📊 性能考虑

### 检索性能
- **Hybrid 检索**: ~100ms
- **Rerank (BGE)**: ~80ms
- **总计**: ~200ms

### 生成性能
- **检索**: ~200ms
- **LLM 生成**: ~1-2s
- **Judge 评估**: ~500ms
- **总计**: ~2-3s

### 优化方向
- [ ] 缓存检索结果（相同 query）
- [ ] 批量 Judge 评估
- [ ] 流式生成
- [ ] 异步预加载 embedding model

## 🔄 扩展路线

### 短期（1-2周）
- [ ] 测试覆盖率 >80%
- [ ] 添加日志和监控
- [ ] 支持更多 Judge 模型（Qwen/Claude）

### 中期（1-2月）
- [ ] 支持流式生成（SSE）
- [ ] 添加缓存层（Redis）
- [ ] 多轮对话上下文管理

### 长期（3-6月）
- [ ] 支持更多检索策略（ColBERT, Dense Passage Retrieval）
- [ ] 自适应拒答阈值（根据历史反馈）
- [ ] 多模态支持（图片、表格）

## 📝 维护建议

### 代码组织
- **新增 MCP Tool**: 在 mcp_server/server.py 添加
- **新增业务逻辑**: 在 rag_core/ 添加
- **新增基础功能**: 在 iteration8/ 添加

### 测试策略
- **单元测试**: 测试 rag_core 函数
- **集成测试**: 测试 MCP Tool 端到端
- **性能测试**: 监控延迟和准确率

### 版本管理
- rag-mcp 独立版本号
- 记录依赖的 iteration8 版本
- 兼容性测试
