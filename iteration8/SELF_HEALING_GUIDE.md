# Self-Healing RAG System Guide

## 概述

Iteration 8 引入了自愈机制，能够自动发现知识库缺陷并生成待审核文件，等待人工批准后补充到 corpus。

## 工作流程

```
评估运行 → 检测问题 → 生成审核文件 → 人工审核 → 批量通过 → 更新 corpus → 重新评估
```

## 触发条件

自愈机制在以下 3 种情况下触发（可在 `rejection_config.json` 配置）:

1. **检索未命中** (`hit != 1`)
   - 查询未命中正确文档
   - 触发原因: `retrieval_miss`

2. **答案排名过低** (`answer_rank > threshold`)
   - 正确答案排名低于阈值（默认 4）
   - 触发原因: `low_rank_5`（数字是实际排名）

3. **Layer 1 拒答** (低分拒答)
   - rerank 分数过低，触发 Layer 1 拒答
   - 触发原因: `low_score_rejection`

## 配置说明

### rejection_config.json

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
- `auto_deduplicate`: 是否自动去重（基于 query 内容）
- `triggers`: 触发条件配置
  - `hit_not_1`: 是否在 hit != 1 时触发
  - `answer_rank_threshold`: 答案排名阈值
  - `layer1_rejection`: 是否在 Layer 1 拒答时触发

## 审核文件格式

每个审核文件是一个 JSON 文件，格式如下:

```json
{
  "query": "SmartLock-100 的临时密码有效期是多久？",
  "ground_truth": "SmartLock-100 的临时密码有效期为 24 小时...",
  "source": {
    "doc_id": "doc-1",
    "char_start": 150,
    "char_end": 280
  },
  "trigger_reason": "retrieval_miss",
  "rejection_reason": null,
  "status": "pending",
  "created_at": "2026-08-10T20:40:09.123456",
  "query_hash": "31fd8078"
}
```

## 使用方法

### 1. 查看待审核项

**方法 A: GitHub Pages 报告**
```bash
# 生成报告
cd iteration8
python generate_report.py --data-dir ../data --output-dir ../docs

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

### 2. 编辑审核内容（可选）

在 GitHub Pages 的 Review 标签页:
- 点击问题文本可以编辑
- 点击答案文本可以编辑
- 编辑后的内容目前不会自动保存（需要手动修改 JSON 文件）

或直接编辑 review 目录下的 JSON 文件。

### 3. 批量通过审核

**方法 A: 使用脚本**
```bash
# 通过指定文件
python approve_reviews.py review_31fd8078_20260810_204009.json

# 通过所有待审核文件
python approve_reviews.py --all

# 指定 corpus 目录
python approve_reviews.py --all --corpus-dir corpus
```

**方法 B: 在 GitHub Pages UI 中**
1. 勾选要通过的审核项
2. 点击 "Approve Selected" 按钮
3. 按照提示执行命令行脚本

### 4. 重新运行评估

通过审核后，需要重新运行评估以查看效果:

```bash
python run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode vector \
  --rerank-mode none \
  --judge-mode deepseek \
  --rejection-preset aggressive \
  --output-dir ../data
```

## GitHub Actions 集成

GitHub Actions 工作流会自动:
1. 运行评估
2. 检测问题并生成审核文件
3. 自动去重
4. 提交审核文件到 review 目录
5. 生成带有审核项的 HTML 报告

### 手动触发评估

在 GitHub 仓库页面:
1. 进入 "Actions" 标签
2. 选择 "RAG Evaluation Pipeline"
3. 点击 "Run workflow"
4. 选择配置参数
5. 点击 "Run workflow"

### 查看审核项

1. 评估完成后，访问 GitHub Pages: `https://<username>.github.io/<repo>`
2. 切换到 "Review" 标签
3. 查看待审核项

### 批量通过（待实现）

未来可以通过 GitHub Actions 实现一键批量通过:
1. 在 UI 中选择要通过的审核项
2. 触发 GitHub Actions workflow
3. 自动更新 corpus 并重新评估

## 文件结构

```
iteration8/
├── self_healing.py           # 自愈核心逻辑
├── approve_reviews.py        # 批量审核脚本
├── generate_report.py        # 报告生成（带审核 UI）
├── rejection_config.json     # 拒答和自愈配置
└── review/                   # 审核文件目录
    ├── .gitkeep
    ├── review_31fd8078_20260810_204009.json
    ├── review_34743fa8_20260810_204009.json
    └── ...
```

## API 参考

### self_healing.py

**主要函数:**

- `should_trigger_self_healing(result, config) -> (bool, str)`
  - 判断是否触发自愈
  - 返回: (是否触发, 触发原因)

- `create_review_file(query, ground_truth, ...) -> str`
  - 创建审核文件
  - 返回: 文件路径

- `deduplicate_review_files(review_dir) -> int`
  - 去重审核文件
  - 返回: 删除的文件数

- `list_pending_reviews(review_dir) -> List[Dict]`
  - 列出待审核文件
  - 返回: 审核文件列表

- `approve_reviews(review_files, corpus_dir) -> Dict`
  - 批量通过审核
  - 返回: 审核结果统计

### approve_reviews.py

**命令行参数:**

```bash
python approve_reviews.py [files...] [--all] [--review-dir DIR] [--corpus-dir DIR]
```

- `files`: 要通过的审核文件名
- `--all`: 通过所有待审核文件
- `--review-dir`: 审核文件目录（默认: review）
- `--corpus-dir`: corpus 目录（默认: corpus）

## 最佳实践

1. **定期检查审核项**
   - 每次评估后检查 GitHub Pages 的 Review 标签
   - 及时处理待审核项

2. **审核前验证内容**
   - 确认 ground_truth 答案准确
   - 检查答案是否来自正确的文档

3. **批量操作**
   - 对于相似的问题，可以一次性批量通过
   - 使用 `--all` 标志批量处理

4. **监控效果**
   - 通过审核后重新运行评估
   - 对比 Hit Rate 和 Rejection Rate 的变化

5. **避免重复**
   - 自动去重功能会保留最新的审核文件
   - 相同 query 只会保留一份

## 故障排除

### 审核文件未生成

检查 `rejection_config.json`:
```json
{
  "self_healing": {
    "enabled": true  // 确保是 true
  }
}
```

### 审核后效果不明显

- 检查 corpus 文件是否更新
- 确认 ground_truth 内容是否正确添加
- 重新运行评估，确保使用更新后的 corpus

### 去重删除了错误的文件

- 检查 query_hash 是否正确
- 查看审核文件的 `created_at` 时间戳
- 手动恢复被删除的文件（如果有备份）

## 未来改进

- [ ] 在 UI 中直接保存编辑的内容
- [ ] 支持在 GitHub Pages 中一键批量通过
- [ ] 添加审核历史记录
- [ ] 支持审核文件的版本控制
- [ ] 自动化 A/B 测试（通过前后对比）

## 相关文档

- [Iteration 8 设计文档](../docs/iteration8-自愈.md)
- [RAG MCP Server](../docs/RAG-MCP.md)
- [迭代计划](../docs/iteration-plan.md)
