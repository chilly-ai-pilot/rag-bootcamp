# RAG Bootcamp: 从零到生产级 RAG 系统

[![Evaluation Status](https://github.com/YOUR_USERNAME/rag-bootcamp/actions/workflows/eval_pipeline.yml/badge.svg)](https://github.com/YOUR_USERNAME/rag-bootcamp/actions)
[![GitHub Pages](https://img.shields.io/badge/Report-GitHub%20Pages-blue)](https://YOUR_USERNAME.github.io/rag-bootcamp/)

一个完整的 RAG (Retrieval-Augmented Generation) 系统构建教程，从基础原型到生产级系统的迭代式开发。

## 🎯 项目目标

通过 7 次迭代，逐步构建一个**可维护、可评估、可优化**的生产级 RAG 系统。

## 📚 迭代概览

| Iteration | 主题 | 核心技术 | 状态 |
|-----------|------|----------|------|
| [0](iteration0/) | **基础原型** | Random Retrieval + GPT Generation | ✅ 完成 |
| [1](iteration1/) | **向量检索** | Embedding + ChromaDB | ✅ 完成 |
| [2](iteration2/) | **Chunking 优化** | Fixed-size vs Semantic Chunking | ✅ 完成 |
| [3](iteration3/) | **检索策略对比** | Vector vs BM25 vs Hybrid | ✅ 完成 |
| [4](iteration4/) | **Reranker** | BGE-Reranker + 两阶段检索 | ✅ 完成 |
| [5](iteration5/) | **LLM-as-Judge** | Faithfulness + Relevance 自动评估 | ✅ 完成 |
| [6](iteration6/) | **拒答机制** | 多层拒答 + 阈值配置 | ✅ 完成 |
| [7](iteration7/) | **持续评估闭环** | GitHub Actions + 自动化 CI/CD | ✅ 完成 |

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 必需的 API Key: `DEEPSEEK_API_KEY`

### 安装依赖

```bash
cd iteration7
pip install -r requirements.txt
```

### 运行评估

```bash
# 本地评估
python run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode hybrid \
  --rerank-mode bge \
  --judge-mode deepseek

# 生成报告
python generate_report.py --data-dir ../data --output-dir ../reports
```

### 查看报告

打开 `reports/index.html` 或访问 [GitHub Pages](https://YOUR_USERNAME.github.io/rag-bootcamp/)。

## 📊 系统架构

```
用户查询
    ↓
[Chunking] 文档切片（fixed_100_50 / semantic）
    ↓
[Retrieval] 召回（Vector / BM25 / Hybrid）
    ↓
[Rerank] 重排序（BGE-Reranker）
    ↓
[Generation] 生成答案（DeepSeek + Citations）
    ↓
[Judge] 质量评估（Faithfulness + Relevance）
    ↓
[Rejection] 拒答判断（Multi-layer）
    ↓
最终答案
```

## 🎓 核心特性

### 1. 多策略检索

- **Vector Retrieval**: 基于语义相似度
- **BM25**: 基于词频统计
- **Hybrid**: 结合两者优势

### 2. 两阶段检索

- **Stage 1 (Recall)**: 召回 40 个候选
- **Stage 2 (Rerank)**: BGE-Reranker 精排到 5 个

### 3. 引用验证

生成的答案自动标注引用源，并验证准确性：
```
根据产品手册[1]，该功能支持批量操作。具体步骤请参考[2]。

[1] doc-3.txt:150-280
[2] doc-5.txt:420-550
```

### 4. LLM-as-Judge 自动评估

- **Faithfulness**: 答案是否忠实于检索内容
- **Answer Relevance**: 答案是否回答用户问题
- **Hit Rate**: 检索是否命中正确文档

### 5. 多层拒答机制

| Layer | 判断依据 | 阈值示例 |
|-------|---------|---------|
| Layer 0 | 检索命中率 | 需要 hit=1 |
| Layer 1 | Rerank 分数 | top1 > 0.5 |
| Layer 2 | 检索-生成一致性 | Citation 覆盖率 > 60% |
| Layer 3 | Judge 评分 | Faithfulness > 0.8 |

### 6. 持续评估 CI/CD

- **自动触发**: corpus 文件变更时自动运行评估
- **结果持久化**: 保存到 `data/` 目录，可追溯
- **可视化报告**: 自动生成 HTML 报告并部署到 GitHub Pages

## 📈 性能指标

最新评估结果（截至 Iteration 7）：

| 指标 | 数值 | 说明 |
|------|------|------|
| **Hit Rate** | ~85% | 检索召回准确率 |
| **Faithfulness** | ~0.88 | 答案忠实度 |
| **Answer Relevance** | ~0.85 | 答案相关性 |
| **Rejection Rate** | ~12% | 拒答率（适度） |
| **MRR** | ~0.75 | 平均倒数排名 |

查看完整报告：[GitHub Pages](https://YOUR_USERNAME.github.io/rag-bootcamp/)

## 🛠️ 技术栈

| 组件 | 技术选型 |
|------|---------|
| **Embedding** | OpenAI text-embedding-3-small |
| **Vector DB** | ChromaDB |
| **BM25** | Rank-BM25 |
| **Reranker** | BAAI/bge-reranker-base |
| **LLM** | DeepSeek-Chat |
| **Judge** | DeepSeek-Chat (可选 Qwen) |
| **CI/CD** | GitHub Actions |
| **可视化** | Chart.js + HTML |

## 📁 项目结构

```
rag-bootcamp/
├── iteration0/          # 基础原型
├── iteration1/          # 向量检索
├── iteration2/          # Chunking 优化
├── iteration3/          # 检索策略对比
├── iteration4/          # Reranker
├── iteration5/          # LLM-as-Judge
├── iteration6/          # 拒答机制
├── iteration7/          # 持续评估闭环 ⭐️
│   ├── chunking.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── scoring.py
│   ├── evaluation.py
│   ├── run_eval.py
│   ├── generate_report.py      # 报告生成器
│   ├── expand_testset.py       # 测试集扩充工具
│   ├── rejection_config.json   # 拒答配置
│   ├── corpus/                 # 文档语料库
│   └── README.md
├── data/                # 评估结果历史
├── reports/             # HTML 报告（GitHub Pages）
├── .github/workflows/   # CI/CD 配置
└── docs/                # 设计文档
```

## 🔧 配置与部署

### 1. 设置 API Key

在仓库 Settings → Secrets 中添加：
```
DEEPSEEK_API_KEY=your_api_key_here
```

### 2. 启用 GitHub Actions

Push 代码到 `main` 分支后，Actions 自动运行。

### 3. 配置 GitHub Pages

Settings → Pages → Source:
- Branch: `main`
- Folder: `/reports`

### 4. 扩充测试集

```bash
cd iteration7

# 查看当前统计
python expand_testset.py --action stats

# 创建新文档
python expand_testset.py --action create-doc --doc-id 8

# 验证格式
python expand_testset.py --action validate
```

## 📖 学习路径

### 新手入门

1. 阅读 [Iteration 0](iteration0/README.md) - 了解 RAG 基础
2. 运行 [Iteration 1](iteration1/README.md) - 体验向量检索
3. 学习 [评估指标](docs/RAG-Cognition.md)

### 进阶优化

4. [Iteration 2](iteration2/README.md) - Chunking 策略对比
5. [Iteration 3](iteration3/README.md) - 检索策略选择
6. [Iteration 4](iteration4/README.md) - Reranker 提升精度

### 生产就绪

7. [Iteration 5](iteration5/README.md) - 自动化评估
8. [Iteration 6](iteration6/README.md) - 拒答机制
9. [Iteration 7](iteration7/README.md) - 持续评估 CI/CD

## 🎯 最佳实践

### Chunking

推荐配置：`fixed_100_50`
- Chunk size: 100 tokens
- Overlap: 50 tokens
- 平衡召回率和精度

### Retrieval

推荐配置：`hybrid + rerank`
- Stage 1: Hybrid (Vector + BM25), top-40
- Stage 2: BGE-Reranker, top-5
- 最佳的召回和精度平衡

### Rejection

推荐配置：`moderate` preset
- Layer 1: Rerank top1 > 0.5
- Layer 3: Faithfulness > 0.8, Relevance > 0.75
- 适度的拒答率（~10-15%）

## 🚨 故障排除

### 问题 1: ChromaDB 连接失败

**解决方案**: 删除 `chroma_db/` 目录，重新初始化。

### 问题 2: API 调用超时

**解决方案**: 降低 `--batch-size` 参数（默认 5，可降至 3）。

### 问题 3: GitHub Actions 失败

**解决方案**: 检查 `DEEPSEEK_API_KEY` 是否正确配置。

### 问题 4: 报告无法访问

**解决方案**: 
1. 确认 GitHub Pages 已启用
2. 等待 1-2 分钟让部署完成
3. 检查 `reports/index.html` 是否存在

## 📚 扩展阅读

- [设计文档](docs/iteration-plan.md)
- [RAG 认知框架](docs/RAG-Cognition.md)
- [测试集扩充指南](iteration7/TESTSET_EXPANSION_GUIDE.md)

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

建议贡献方向：
- 新的 Chunking 策略
- 更多 Retrieval 算法
- 优化 Reranker 性能
- 扩充测试集
- 改进可视化报告

## 📄 许可证

MIT License

## 🙏 致谢

感谢以下项目和资源：
- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [BGE-Reranker](https://github.com/FlagOpen/FlagEmbedding)
- [DeepSeek](https://www.deepseek.com/)

## 📞 联系方式

- Issues: [GitHub Issues](https://github.com/YOUR_USERNAME/rag-bootcamp/issues)
- Email: your.email@example.com

---

**⭐️ 如果这个项目对你有帮助，请给一个 Star！**

**Built with ❤️ for the RAG community**