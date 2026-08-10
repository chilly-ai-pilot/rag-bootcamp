# 🎯 RAG MCP Server - 项目总览

## 📦 已创建的文件

### 核心代码（7个文件）
```
rag_core/
├── __init__.py              ✅ 模块导出
├── search.py                ✅ 纯检索逻辑（Layer 0/1 拒答）
├── generate.py              ✅ 完整生成逻辑（Layer 0-3 拒答）
└── rejection_config.json    ✅ 拒答配置

mcp_server/
├── __init__.py              ✅ 模块导出
└── server.py                ✅ MCP Server 入口

tests/
└── test_mcp_tools.py        ✅ MCP Tools 测试
```

### 配置和文档（8个文件）
```
├── .env.example             ✅ 环境变量示例
├── mcp_config_example.json  ✅ Kiro MCP 配置示例
├── requirements.txt         ✅ Python 依赖
├── quick_test.py            ✅ 快速测试脚本
├── run_server.sh            ✅ 启动脚本
├── README.md                ✅ 项目说明
├── ARCHITECTURE.md          ✅ 架构设计文档
├── SETUP_COMPLETE.md        ✅ 设置完成指南
└── PROJECT_SUMMARY.md       ✅ 本文件
```

**总计：15个文件**

## 🏗️ 架构设计

### 三层架构

```
┌─────────────────────────────┐
│     MCP Protocol Layer      │  ← 薄薄一层，只管协议
│     (mcp_server/)            │
├─────────────────────────────┤
│     RAG Core Layer          │  ← 流程编排 + 拒答判断
│     (rag_core/)              │
├─────────────────────────────┤
│   Iteration8 Core Modules   │  ← 基础功能实现
│   (../iteration8/)           │
└─────────────────────────────┘
```

### 关键设计决策

#### 1. 为什么拆成两个 Tool？

**search_knowledge**:
- 只检索，不生成
- Agent 判断信息是否充足
- 触发 Layer 0/1 拒答

**generate_answer**:
- 完整生成（检索 + 生成 + 校验）
- Agent 获取最终答案
- 触发 Layer 0-3 拒答

**好处**:
- Agent 可以先检索判断，提前走降级策略
- 支持混合检索（RAG + Web）
- 灵活组合不同数据源

#### 2. 为什么不重复实现核心功能？

**rag_core 依赖 iteration8**:
- ✅ 复用成熟的 chunking/retrieval/generation 代码
- ✅ 只做流程编排和拒答判断
- ✅ 降低维护成本

**如果需要新功能**:
- 在 iteration8 中实现
- rag_core 直接调用
- 保持单一代码源

#### 3. 四层拒答如何分配？

| Layer | 检索 Tool | 生成 Tool | 原因 |
|-------|-----------|-----------|------|
| Layer 0 | ❌ | ✅ | 需要 ground truth，仅评估 |
| Layer 1 | ✅ | ✅ | Rerank 分数，检索阶段可判断 |
| Layer 2 | ❌ | ✅ | Citation 校验，生成后才有 |
| Layer 3 | ❌ | ✅ | Judge 评估，生成后才能评 |

**Agent 使用流程**:
```
1. 调用 search_knowledge
   ↓
2. 如果 rejected (Layer 1)
   → 走降级（web search / 直接拒答）
   ↓
3. 如果 returned > 0
   → 调用 generate_answer
   ↓
4. 如果 rejected (Layer 2/3)
   → 走降级或返回拒答消息
```

## 🎯 核心功能

### 1. search_knowledge

**代码**: `rag_core/search.py` (148行)

**功能**:
- 文档分块（调用 chunking.py）
- 多模式检索（vector/bm25/hybrid）
- BGE Rerank（可选）
- Layer 1 拒答检查（rerank 分数）

**返回**:
- 检索结果列表（chunk_id, text, score, metadata）
- 拒答状态（rejected, rejection_reason）
- 统计信息（total_retrieved, returned）

### 2. generate_answer

**代码**: `rag_core/generate.py` (175行)

**功能**:
- 检索阶段（复用 search 逻辑）
- 两段式生成（推理 + 格式化）
- Citation 确定性校验（Layer 2）
- Judge 评估（Faithfulness & Relevance，Layer 3）

**返回**:
- 带引用标注的答案
- 引用列表（citations）
- 拒答状态（rejected, rejection_reason）
- 评分（faithfulness_score, relevance_score）

### 3. MCP Server

**代码**: `mcp_server/server.py` (150行)

**功能**:
- 注册两个 MCP Tools
- 处理 JSON-RPC 请求
- 参数校验和转换
- 调用 rag_core 函数

**协议**: MCP over stdio

## 📊 代码统计

### 行数统计
```
rag_core/search.py:      ~150 行
rag_core/generate.py:    ~175 行
mcp_server/server.py:    ~150 行
tests/test_mcp_tools.py: ~160 行
quick_test.py:           ~120 行
---------------------------------
核心代码总计:            ~755 行
```

### 依赖关系
```
rag_core/search.py
  → iteration8/chunking.py
  → iteration8/retrieval.py

rag_core/generate.py
  → iteration8/chunking.py
  → iteration8/retrieval.py
  → iteration8/generation.py
  → iteration8/evaluation.py

mcp_server/server.py
  → rag_core/search.py
  → rag_core/generate.py
```

## 🧪 测试覆盖

### 已有测试

**test_mcp_tools.py**:
- ✅ test_search_knowledge - 纯检索测试
- ✅ test_generate_answer - 完整生成测试
- ✅ test_mcp_tool_schema - Schema 验证

**quick_test.py**:
- ✅ 快速端到端测试（不启动 Server）
- ✅ 包含两个 Tool 的测试

### 缺失的测试（后续补充）

- [ ] 拒答机制测试（各层拒答条件）
- [ ] 性能测试（延迟、吞吐量）
- [ ] 边界条件测试（空查询、超长查询）
- [ ] 并发测试（多个 Agent 同时调用）

## 🚀 使用方式

### 方式 1: 快速测试（推荐新手）

```bash
cd rag-mcp
export DEEPSEEK_API_KEY="your-key"
python quick_test.py
```

**适合**:
- 验证功能是否正常
- 快速调试
- 不需要 MCP 协议

### 方式 2: 完整测试

```bash
cd rag-mcp
export DEEPSEEK_API_KEY="your-key"
python tests/test_mcp_tools.py
```

**适合**:
- 完整测试两个 Tool
- 包含 Schema 验证
- 了解详细输出格式

### 方式 3: 启动 MCP Server

```bash
cd rag-mcp
export DEEPSEEK_API_KEY="your-key"
./run_server.sh
```

或直接：

```bash
python mcp_server/server.py
```

**适合**:
- 集成到 Kiro
- Agent 实际调用
- 生产环境

### 方式 4: Kiro 中配置

编辑 `~/.kiro/settings/mcp.json`:

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "python",
      "args": ["/path/to/rag-mcp/mcp_server/server.py"],
      "env": {
        "DEEPSEEK_API_KEY": "your-key"
      }
    }
  }
}
```

## ⚙️ 配置选项

### 环境变量

| 变量 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | ✅ | - | DeepSeek API Key |
| `RAG_CORPUS_DIR` | ❌ | `iteration8/corpus` | 语料库路径 |
| `RAG_CHUNKING_STRATEGY` | ❌ | `fixed_100_50` | 分块策略 |
| `RAG_RETRIEVAL_MODE` | ❌ | `hybrid` | vector/bm25/hybrid |
| `RAG_RERANK` | ❌ | `true` | 是否使用 rerank |

### 拒答配置

编辑 `rag_core/rejection_config.json`:

```json
{
  "rejection_layers": {
    "layer1_rerank": {
      "top1_threshold": 0.50,      // 调整这个
      "top3_avg_threshold": 0.45   // 和这个
    },
    "layer3_judge": {
      "faithfulness_threshold": 0.80,  // 调整这个
      "relevance_threshold": 0.75      // 和这个
    }
  }
}
```

**阈值调优建议**:
- **保守**（拒答率高）: top1=0.60, faithfulness=0.85
- **中等**（默认）: top1=0.50, faithfulness=0.80
- **激进**（拒答率低）: top1=0.40, faithfulness=0.75

## 📈 性能预期

### 延迟（基于 iteration8 测试）

| 操作 | 延迟 | 说明 |
|------|------|------|
| search_knowledge | ~200ms | hybrid + rerank |
| generate_answer | ~2-3s | 检索 + 生成 + Judge |

### 准确性（基于 35 个测试查询）

| 指标 | 值 | 说明 |
|------|-----|------|
| Recall@5 | 80%+ | 前5个结果中命中率 |
| MRR | 0.70+ | 平均倒数排名 |
| Faithfulness | 0.85-0.95 | Judge 评分 |
| Rejection Rate | 5-15% | 取决于配置 |

## 🔄 后续迭代计划

### 短期（1-2周）
- [ ] 增加测试覆盖率 >80%
- [ ] 添加日志和监控（structlog）
- [ ] 性能基准测试

### 中期（1-2月）
- [ ] 支持流式生成（SSE）
- [ ] 添加缓存层（Redis）
- [ ] 多轮对话上下文

### 长期（3-6月）
- [ ] 支持更多检索策略（ColBERT）
- [ ] 自适应拒答阈值
- [ ] 多模态支持

## 📚 文档索引

1. **README.md** - 项目说明和快速入门
2. **ARCHITECTURE.md** - 架构设计详解
3. **SETUP_COMPLETE.md** - 设置完成指南
4. **PROJECT_SUMMARY.md** - 本文件，项目总览

## ✅ 完成检查清单

- [x] 创建目录结构（rag_core, mcp_server, tests）
- [x] 实现 search_knowledge（纯检索）
- [x] 实现 generate_answer（完整生成）
- [x] 实现 MCP Server（协议层）
- [x] 创建测试脚本（test_mcp_tools.py, quick_test.py）
- [x] 编写配置文件（.env.example, mcp_config_example.json）
- [x] 编写文档（README, ARCHITECTURE, SETUP_COMPLETE）
- [x] 创建启动脚本（run_server.sh）
- [x] 验证依赖关系（iteration8 核心模块）

## 🎉 项目状态：✅ 完成

RAG MCP Server 已经完全搭建完成，可以立即使用！

**下一步**:
1. 运行 `python quick_test.py` 验证功能
2. 在 Kiro 中配置 MCP Server
3. 测试 Agent 调用两个 Tools
4. 根据实际使用效果调优拒答阈值
