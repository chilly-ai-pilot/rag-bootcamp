rag-mcp/
├── rag_core/                  # RAG 核心逻辑（你现有的代码）
│   ├── search.py              # 检索函数
│   ├── generate.py            # 生成函数
│   ├── evaluate.py            # 评估脚本（依赖 queries.json）
│   └── queries.json           # 测试集
│
├── mcp_server/                # MCP 协议层（新增，薄薄一层）
│   ├── server.py              # MCP Server 入口，调 rag_core 的函数
│   └── __init__.py
│
└── tests/                     # 可以放一起
    └── test_mcp_tools.py      # 测试 MCP Tool 的调用

暴露两个接口，按检索和生成拆分。

---

## 接口定义

### `search_knowledge` — 纯检索

只检索知识库，不做生成。Agent 用来判断"知识库里有没有相关内容、够不够回答"。

```
输入：
  query:  string   # 用户问题或搜索关键词
  top_k:  int      # 返回片段数，默认 10，最大 20

输出：
  results:          # 检索到的文档片段列表
    - chunk_id
      text
      score         # 相关性分数
      doc_id
      metadata
  total_retrieved:  # 粗筛候选集大小
  returned:          # 实际返回数量
  rejected: bool     # 是否触发检索层拒答
  rejection_reason:  # 拒答原因（layer0/layer1）
```

### `generate_answer` — 完整生成

检索 + 两段式生成 + 确定性校验。Agent 用来获取最终答案。

```
输入：
  query:  string   # 用户问题
  top_k:  int      # 检索片段数，默认 5

输出：
  answer:            # 带引用标注的完整答案
  citations:         # 引用列表
    - span           # 被引用的文字
      source         # 来源标识
      chunk_id
  rejected: bool     # 是否触发拒答
  rejection_reason:  # 拒答原因（layer0-3）
  faithfulness_score # Judge 模型打分
```

---

## 为什么拆成两个

Agent 需要灵活编排，拆开后能先检索判断信息够不够，不够再上网搜，搜完再生成。一个 Tool 做不到这种按需组合。同时和你的四层拒答体系对齐——检索 Tool 触发 layer0/layer1，生成 Tool 触发 layer2/layer3，Agent 在检索阶段就能提前走降级策略。