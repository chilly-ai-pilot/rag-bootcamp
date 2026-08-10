# Iteration 8: Self-Healing RAG System

## 📋 概述

Iteration 8 在 Iteration 7 的基础上增加了**自愈机制**，能够自动发现知识库缺陷并生成待审核文件，经人工批准后补充到 corpus，形成完整的自我改进闭环。

### 核心功能

1. **自动缺陷检测**：在 3 个时间节点检测问题（hit非1、answer_rank>4、low_score_rejection）
2. **审核文件生成**：自动生成待审核的 JSON 文件，记录问题和建议答案
3. **智能去重**：基于 query 内容自动去重，避免重复审核
4. **可视化审核 UI**：HTML 报告中新增 Review 标签，支持批量审核
5. **内容可编辑**：审核页面支持编辑问题和答案
6. **GitHub Actions 集成**：自动提交审核文件，形成完整闭环

## 🚀 快速开始

### 1. 本地运行评估（自动触发自愈）

```bash
cd iteration8

# 标准评估（带自愈检查）
python run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode vector \
  --rerank-mode none \
  --judge-mode deepseek \
  --rejection-preset aggressive \
  --output-dir ../data
```

评估完成后会自动：
- 检测问题并生成审核文件到 `review/` 目录
- 自动去重（保留最新的）
- 输出统计信息（触发次数、待审核数）

### 2. 查看待审核项

**方法 A: HTML 报告（推荐）**

```bash
# 生成带审核 UI 的报告
python generate_report.py --data-dir ../data --output-dir ../docs --review-dir review

# 在浏览器中打开 docs/index.html
# 切换到 "Review" 标签页
```

**方法 B: 命令行**

```python
from self_healing import list_pending_reviews

pending = list_pending_reviews("review")
for review in pending:
    print(f"Query: {review['query']}")
    print(f"Reason: {review['trigger_reason']}")
```

### 3. 批量通过审核

```bash
# 通过指定文件
python approve_reviews.py review_31fd8078_20260810_204009.json

# 通过所有待审核文件
python approve_reviews.py --all

# 指定目录
python approve_reviews.py --all --review-dir review --corpus-dir corpus
```

### 4. 重新评估查看效果

```bash
python run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode vector \
  --output-dir ../data
```

## 🔍 自愈触发条件

### 1. 检索未命中 (`retrieval_miss`)

**触发条件**: `hit != 1`
- 查询未命中正确的文档
- 说明知识库中可能缺少相关内容

**示例**:
```json
{
  "trigger_reason": "retrieval_miss",
  "query": "SmartCam-200 的夜视距离是多少？",
  "ground_truth": "SmartCam-200 配备红外夜视功能，夜视距离可达 30 米..."
}
```

### 2. 答案排名过低 (`low_rank_N`)

**触发条件**: `answer_rank > threshold`（默认 4）
- 正确答案排名太靠后
- 说明检索相关性不足

**示例**:
```json
{
  "trigger_reason": "low_rank_7",
  "query": "SmartLock-100 如何生成临时密码？",
  "ground_truth": "SmartLock-100 支持通过手机 App..."
}
```

### 3. Layer 1 拒答 (`low_score_rejection`)

**触发条件**: rerank 分数过低
- 检索结果相关性分数低于阈值
- Layer 1 拒答机制触发

**示例**:
```json
{
  "trigger_reason": "low_score_rejection",
  "rejection_reason": "Layer 1: 检索结果相关性不足 (最高分: 0.23)",
  "query": "智能家居设备如何联网？",
  "ground_truth": "所有设备支持 Wi-Fi 和蓝牙..."
}
```

## ⚙️ 配置说明

### rejection_config.json

自愈机制的配置在 `rejection_config.json` 中:

```json
{
  "self_healing": {
    "enabled": true,
    "review_dir": "review",
    "auto_deduplicate": true,
    "triggers": {
      "hit_not_1": true,
      "answer_rank_threshold": 4,
      "layer1_rejection": true
    }
  }
}
```

**参数说明:**
- `enabled`: 是否启用自愈机制
- `review_dir`: 审核文件存放目录
- `auto_deduplicate`: 是否自动去重（基于 query 内容 hash）
- `triggers`:
  - `hit_not_1`: 在 hit != 1 时触发
  - `answer_rank_threshold`: 答案排名阈值（默认 4）
  - `layer1_rejection`: 在 Layer 1 拒答时触发

## 📊 审核 UI 功能

HTML 报告新增 **Review 标签页**，提供以下功能:

1. **审核列表**: 显示所有待审核项
   - 查询问题
   - 触发原因（带颜色标签）
   - Ground truth 答案
   - 创建时间和来源文档

2. **内容编辑**: 
   - 点击问题文本可编辑
   - 点击答案文本可编辑
   - （注: 当前版本编辑不会自动保存，需手动修改 JSON 文件）

3. **批量操作**:
   - Select All / Deselect All
   - Approve Selected (批量通过)

4. **空状态提示**:
   - 无待审核项时显示友好提示

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
3. Branch 选择 `main`，目录选择 `/docs`
4. 保存后，GitHub Pages 将自动部署
5. 访问 `https://<username>.github.io/<repo-name>/` 查看报告

## 🔄 工作流触发条件

### 自动触发

以下文件变更时会自动触发评估：

- `iteration8/corpus/doc-*.txt` - 新增或修改文档
- `iteration8/corpus/queries.json` - 修改查询集
- `iteration8/*.py` - 修改评估代码
- `iteration8/rejection_config.json` - 修改拒答配置

评估完成后会自动：
1. 检测问题并生成审核文件
2. 去重审核文件
3. 提交审核文件到 `iteration8/review/` 目录
4. 生成带审核 UI 的 HTML 报告
5. 提交结果到 `data/` 和 `docs/` 目录

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

### HTML 报告（`docs/` 目录）

```
docs/
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
- [ ] `iteration8/review/` 目录生成审核文件（如果有问题被检测到）
- [ ] `docs/index.html` 更新，包含 Review 标签页
- [ ] 通过 GitHub Pages 可访问报告
- [ ] Review 标签页显示待审核项数量
- [ ] 可以在 UI 中编辑问题和答案
- [ ] 运行 `approve_reviews.py --all` 能批量通过审核
- [ ] 通过审核后重新评估，指标有改善

## 🔄 自愈工作流

```
评估运行
  ↓
检测问题（3个触发条件）
  ↓
生成审核文件 (review/*.json)
  ↓
自动去重（基于 query hash）
  ↓
GitHub Actions 提交审核文件
  ↓
生成 HTML 报告（带 Review UI）
  ↓
人工审核（GitHub Pages）
  ↓
批量通过审核
  ↓
更新 corpus
  ↓
重新评估
  ↓
查看改善效果
```

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
2. 确认 Branch 设置为 `main`，目录设置为 `/docs`
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

1. **UI 中直接保存编辑**: 在 HTML 页面编辑后直接保存到 JSON 文件
2. **一键批量通过**: 在 GitHub Pages 中直接触发 GitHub Actions 批量通过
3. **审核历史**: 记录所有审核操作的历史
4. **A/B 测试**: 对比通过前后的指标变化
5. **自动化建议**: 使用 LLM 生成更好的 ground truth 答案
6. **智能优先级**: 根据问题严重程度排序待审核项

## 📚 详细文档

- [Self-Healing 使用指南](./SELF_HEALING_GUIDE.md) - 完整的自愈系统使用文档
- [Iteration 8 设计文档](../docs/iteration8-自愈.md) - 架构设计和实现细节
- [RAG MCP Server](../docs/RAG-MCP.md) - MCP 服务器文档

## 📞 联系方式

如有问题，请提 Issue 或联系维护者。

---

**Happy Self-Healing! 🔄🚀**
