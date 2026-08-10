# 自愈审核批准指南

## 📋 概述

当 RAG 系统检测到知识库不足以回答某些问题时，会自动触发自愈机制，生成待审核的 QA 对。你可以在 GitHub Pages 报告中查看这些待审核内容，并通过 GitHub Actions 一键批准。

## 🚀 如何批准审核

### 方法 1: 通过 GitHub Pages 报告（推荐）

1. **查看报告**
   - 访问: https://chilly-ai-pilot.github.io/rag-bootcamp/
   - 点击 "🔍 Review" 标签页

2. **审查 QA 对**
   - 查看所有待审核的问题和答案
   - 你可以直接在网页上编辑问题和答案（仅用于预览）

3. **批准审核**
   - 点击 "🚀 Approve via GitHub Actions" 按钮
   - 自动跳转到 GitHub Actions 页面
   - 点击右上角的 "Run workflow" 按钮
   - 选择 "yes" 选项
   - 点击绿色的 "Run workflow" 按钮

4. **等待完成**
   - Workflow 会自动运行（约 1-2 分钟）
   - 完成后，所有待审核的 QA 对会被添加到知识库
   - 新文档会以 `doc-N.txt` 的形式创建在 `iteration8/corpus/` 目录下

### 方法 2: 通过命令行（本地操作）

如果你需要先修改 QA 对内容再批准：

1. **编辑审核文件**
   ```bash
   cd iteration8/review
   # 编辑对应的 review_*.json 文件
   ```

2. **批准审核**
   ```bash
   cd iteration8
   
   # 批准所有待审核
   python approve_reviews.py --all
   
   # 或批准特定文件
   python approve_reviews.py review_abc123_20260810_120000.json
   ```

3. **提交更改**
   ```bash
   git add .
   git commit -m "✅ Approved reviews and added new documents"
   git push
   ```

## 📊 审核后会发生什么

1. **创建新文档**: 每个 QA 对会生成一个新文档 `doc-N.txt`，格式如下：
   ```
   问：[问题内容]
   
   答：[答案内容]
   ```

2. **更新审核状态**: `review_*.json` 文件的状态从 `pending` 改为 `approved`

3. **重新评估**: 运行 eval_pipeline workflow 可以看到改进效果

## ⚙️ GitHub Actions Workflow

### approve_reviews.yml

这个 workflow 支持手动触发（`workflow_dispatch`），功能：

- ✅ 自动批准所有待审核文件
- 📝 将 QA 对添加到 corpus
- 🔄 自动提交和推送更改
- 📊 显示审核统计信息

### 触发方式

1. 进入仓库的 Actions 页面
2. 选择 "Approve Reviews" workflow
3. 点击 "Run workflow"
4. 选择 "yes" 确认批准
5. 点击绿色的 "Run workflow" 按钮

## 💡 注意事项

### 网页编辑限制

- ⚠️ 网页上的内联编辑**仅用于预览**，不会保存
- 如需修改 QA 对内容，请：
  1. 编辑 `iteration8/review/review_*.json` 文件
  2. 然后使用命令行批准，或
  3. 提交修改后再使用 GitHub Actions 批准

### 文档命名

- 新文档会自动按序号命名：`doc-17.txt`, `doc-18.txt` 等
- 编号从现有最大编号+1 开始
- 不会覆盖已有文档

### 审核触发条件

自愈机制在以下情况触发：

1. **hit非1**: 检索未命中正确文档
2. **answer_rank>4**: 正确答案排名太靠后（>4）
3. **low_score_rejection**: Rerank 分数过低

## 🔄 完整流程示例

```bash
# 1. 系统自动运行评估（GitHub Actions）
#    → 检测到 12 个问题无法正确回答
#    → 生成 12 个 review_*.json 文件

# 2. 查看报告
#    → 访问 GitHub Pages
#    → 点击 Review 标签页
#    → 查看 12 个待审核 QA 对

# 3. 批准审核
#    → 点击 "Approve via GitHub Actions"
#    → 在 GitHub Actions 页面运行 workflow
#    → 等待 1-2 分钟

# 4. 验证结果
#    → 检查 iteration8/corpus/ 目录
#    → 新增了 12 个 doc-*.txt 文件
#    → review/*.json 文件状态变为 approved

# 5. 重新评估
#    → 运行 eval_pipeline workflow
#    → 观察 Hit Rate 是否提升
```

## 📈 效果评估

批准审核后，你应该能看到：

- **Hit Rate** ↑: 之前漏检的问题现在能正确检索
- **MRR Score** ↑: 平均排名提升
- **Rejection Rate** ↓: 拒答率降低

运行新的评估来验证改进：

```bash
cd iteration8
python run_eval.py
```

或通过 GitHub Actions 自动运行。

## 🆘 常见问题

### Q: 为什么网页上的编辑不保存？

A: 这是设计决定。复杂的编辑建议在 JSON 文件中完成，确保格式正确。网页编辑仅用于快速预览调整。

### Q: 可以只批准部分审核吗？

A: 目前 GitHub Actions workflow 是批准所有待审核。如需选择性批准，请使用命令行：

```bash
python approve_reviews.py review_abc123.json review_def456.json
```

### Q: 批准后可以撤销吗？

A: 可以，但需要手动操作：

1. 删除对应的 `doc-N.txt` 文件
2. 将 `review_*.json` 的 status 改回 `pending`
3. 提交更改

### Q: Workflow 失败了怎么办？

A: 检查 Actions 日志，常见原因：

- `DEEPSEEK_API_KEY` 未配置
- 依赖安装失败
- 合并冲突

---

📚 更多信息请参考：
- [SELF_HEALING_GUIDE.md](../iteration8/SELF_HEALING_GUIDE.md)
- [GitHub Actions 文档](.github/workflows/)
