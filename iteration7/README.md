# Iteration 7: 持续评估闭环

## 📋 概述

Iteration 7 实现了完整的 CI/CD 评估流程，将手动评估升级为自动化的持续评估系统。

### 核心功能

1. **GitHub Actions 自动触发**：corpus 目录有新文档时自动运行评估
2. **结果持久化**：评估结果保存到 `data/` 目录，版本可追溯
3. **可视化报告**：生成交互式 HTML 报告，通过 GitHub Pages 访问
4. **趋势分析**：追踪指标随时间的变化（Faithfulness、Relevance、Hit Rate 等）

## 🚀 快速开始

### 1. 本地运行评估

```bash
cd iteration7

# 标准评估（输出到当前目录）
python run_eval.py --chunking-strategy fixed_100_50 --retrieval-mode hybrid --rerank-mode bge

# 指定输出目录（用于 CI/CD）
python run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode hybrid \
  --rerank-mode bge \
  --judge-mode deepseek \
  --rejection-preset moderate \
  --output-dir ../data
```

### 2. 生成 HTML 报告

```bash
cd iteration7

# 从 data/ 读取所有结果，生成报告到 reports/
python generate_report.py --data-dir ../data --output-dir ../reports

# 使用默认路径
python generate_report.py
```

### 3. 查看报告

在浏览器中打开 `reports/index.html`，或通过 GitHub Pages 访问。

## ⚙️ GitHub Actions 配置

### 步骤 1：设置 API 密钥

在仓库的 **Settings → Secrets and variables → Actions** 中添加：

| Secret 名称 | 说明 | 是否必需 |
|------------|------|---------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（用于 Judge 和 Generation） | ✅ **必需** |

**如何获取 DeepSeek API Key:**
1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 注册/登录账号
3. 在 API Keys 页面创建新密钥
4. 复制密钥并添加到 GitHub Secrets

> **注意**: 如果未来需要使用其他模型（如 Qwen、OpenAI），可以添加对应的 Secret：
> - `ALI_API_KEY` + `ALI_BASE_URL` (阿里云 Qwen)
> - `OPENAI_API_KEY` (OpenAI)
> - `ANTHROPIC_API_KEY` (Anthropic Claude)

### 步骤 2：启用 GitHub Actions

1. 确保 `.github/workflows/eval_pipeline.yml` 存在
2. 推送代码到 `main` 分支
3. 在仓库的 Actions 标签页查看工作流

### 步骤 3：配置 GitHub Pages

1. 进入仓库的 Settings → Pages
2. Source 选择 "Deploy from a branch"
3. Branch 选择 `main`，目录选择 `/reports`
4. 保存后，GitHub Pages 将自动部署
5. 访问 `https://<username>.github.io/<repo-name>/` 查看报告

## 🔄 工作流触发条件

### 自动触发

以下文件变更时会自动触发评估：

- `iteration7/corpus/doc-*.txt` - 新增或修改文档
- `iteration7/corpus/queries.json` - 修改查询集
- `iteration7/*.py` - 修改评估代码
- `iteration7/rejection_config.json` - 修改拒答配置

### 手动触发

在 Actions 标签页，选择 "RAG Evaluation Pipeline" → "Run workflow"，可以自定义参数：

- Chunking Strategy（默认：fixed_100_50）
- Retrieval Mode（默认：hybrid）
- Rerank Mode（默认：bge）
- Judge Mode（默认：deepseek）
- Rejection Preset（默认：moderate）

## 📊 输出文件

### 评估结果（`data/` 目录）

```
data/
├── results_fixed_100_50_hybrid_rerank_bge_20260807_143022.json
├── results_fixed_100_50_hybrid_rerank_bge_20260808_091534.json
└── baseline.json  # 第一次运行的结果（手动创建）
```

每个结果文件包含：

- `metadata`: 评估时间、corpus 统计、模型配置
- `scores`: 分类评分（Hit、Recall@K）
- `mrr_scores`: MRR 分数
- `results`: 每个查询的详细结果
- `faithfulness_analysis`: Faithfulness 分析
- `relevance_analysis`: Relevance 分析
- `rerank_score_distribution`: Rerank 分数分布

### HTML 报告（`reports/` 目录）

```
reports/
└── index.html  # 主报告页面
```

报告内容：

- **Summary Cards**: 最新指标（Hit Rate、Faithfulness、Relevance、Rejection Rate、MRR）
- **对比视图**: 与上一次评估的差异
- **配置信息**: Corpus 统计、模型配置
- **趋势图**: 指标随时间变化（Chart.js）
- **历史表格**: 所有评估记录

## 🎯 验收检查清单

- [ ] 在 `corpus/` 下添加 `doc-8.txt`，push 后自动触发 workflow
- [ ] Workflow 在 5 分钟内完成，无错误
- [ ] `data/` 目录生成新的结果 JSON 文件（带时间戳）
- [ ] `reports/index.html` 更新
- [ ] 通过 GitHub Pages 可访问报告
- [ ] HTML 报告显示完整指标和趋势图
- [ ] 修改代码（如改 embedding 模型），能检测到性能变化

## 📈 测试集扩充策略

### 目标

从当前 7 个文档扩充到 20-30 个，查询从 30-50 条扩充到 100+ 条。

### 扩充来源

1. **误判案例**: 从 Iteration 6 的分析中提取被误判的案例
2. **边界情况**: 
   - 极短文档（< 100 字）
   - 超长文档（> 5000 字）
   - 多主题文档
3. **真实案例**: 从实际业务场景中收集

### 扩充步骤

1. 在 `corpus/` 下添加新文档：`doc-8.txt`, `doc-9.txt`, ...
2. 在 `queries.json` 中添加对应的查询
3. Push 到 main 分支，自动触发评估
4. 查看报告，分析新文档的影响

## 🔧 成本控制

### 1. 限制并发

工作流配置了并发限制：

```yaml
concurrency:
  group: evaluation-pipeline
  cancel-in-progress: false
```

同时只运行一个评估实例，避免并发触发。

### 2. 缓存 Embedding

Embedding 结果缓存到 `chroma_db/` 目录：

- 相同文档不重复计算 embedding
- 减少 API 调用次数

### 3. 控制触发频率

只在必要文件变更时触发：

- `corpus/*.txt` - 文档变更
- `queries.json` - 查询变更
- 核心代码变更

避免频繁触发（如 README 更新不触发评估）。

## 🛠️ 故障排除

### 问题 1: Workflow 失败 - API 密钥未设置

**症状**: 
```
ValueError: 请设置环境变量 DEEPSEEK_API_KEY
```

**解决方案**: 
在仓库 Settings → Secrets 中添加 `DEEPSEEK_API_KEY`。

### 问题 2: GitHub Pages 无法访问报告

**症状**: 
404 Not Found

**解决方案**: 
1. 检查 Settings → Pages 是否启用
2. 确认 Branch 设置为 `main`，目录设置为 `/reports`
3. 等待几分钟让 GitHub Pages 部署完成

### 问题 3: 评估时间过长

**症状**: 
Workflow 超过 10 分钟仍未完成

**解决方案**: 
1. 减少 `--batch-size`（默认 5，可以降到 3）
2. 使用 `--judge-mode none` 跳过 Judge 评估（测试用）
3. 减少查询数量（仅评估部分查询）

### 问题 4: 报告趋势图显示不正常

**症状**: 
图表显示断点或数据缺失

**解决方案**: 
1. 检查 `data/` 目录中的结果文件格式是否正确
2. 确保所有结果文件都包含 `metadata` 字段
3. 删除损坏的结果文件，重新运行评估

## 📚 技术细节

### 时间戳格式

文件名使用 `YYYYMMDD_HHMMSS` 格式：

```
results_fixed_100_50_hybrid_rerank_bge_20260807_143022.json
                                         ^^^^^^^^ ^^^^^^
                                         日期      时间
```

### 元数据结构

```json
{
  "metadata": {
    "timestamp": "20260807_143022",
    "evaluation_date": "2026-08-07T14:30:22.123456",
    "corpus_stats": {
      "num_documents": 7,
      "num_queries": 30,
      "corpus_dir": "corpus"
    },
    "model_config": {
      "chunking_strategy": "fixed_100_50",
      "retrieval_mode": "hybrid",
      "rerank_mode": "bge",
      "judge_mode": "deepseek",
      "rejection_enabled": true,
      "rejection_preset": "moderate"
    }
  },
  "config": { ... },
  "scores": { ... },
  "results": [ ... ]
}
```

### Chart.js 配置

使用 CDN 引入（无需本地安装）：

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
```

支持的图表类型：

- **折线图**: Faithfulness、Relevance、MRR 趋势
- **双轴图**: Hit Rate vs Rejection Rate

## 🔄 后续优化方向

1. **A/B 测试**: 同时评估多个配置，选择最优
2. **告警机制**: 指标退化时发送通知（Slack、邮件）
3. **增量评估**: 只评估新增的查询，节省时间
4. **性能优化**: 并行处理、GPU 加速
5. **测试集质量**: 定期清理低质量查询
6. **多环境支持**: 区分开发/测试/生产环境

## 📞 联系方式

如有问题，请提 Issue 或联系维护者。

---

**Happy Evaluating! 🚀**
