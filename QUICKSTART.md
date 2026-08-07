# 🚀 快速启动指南

5 分钟快速体验 RAG Bootcamp！

## ✅ 前置条件

- Python 3.10+
- Git
- DeepSeek API Key（[获取地址](https://platform.deepseek.com/)）

## 📦 步骤 1: 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/rag-bootcamp.git
cd rag-bootcamp
```

## 🔑 步骤 2: 设置 API Key

```bash
# macOS/Linux
export DEEPSEEK_API_KEY="your_api_key_here"

# Windows (PowerShell)
$env:DEEPSEEK_API_KEY="your_api_key_here"

# 或者创建 .env 文件
echo "DEEPSEEK_API_KEY=your_api_key_here" > .env
```

## 📚 步骤 3: 安装依赖

```bash
cd iteration7
pip install -r requirements.txt
```

## 🎯 步骤 4: 运行评估

```bash
# 快速评估（使用默认配置）
python run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode hybrid \
  --rerank-mode bge \
  --judge-mode deepseek
```

预期输出：
```
============================================================
Running evaluation:
  Chunking: fixed_100_50
  Retrieval: hybrid
  Rerank: bge
  Judge Mode: deepseek
============================================================
✅ Rejection mechanism enabled
   Config file: rejection_config.json

Corpus: 35 chunks from 7 docs
Average chunk size: 189.3 characters

============================================================
Batch Retrieve + Generate + Judge (async)
============================================================
🚀 批量生成 35 个查询的答案（滑动窗口并发数：10）...
...
✅ Results written to results_fixed_100_50_hybrid_rerank_bge_20260808_143022.json
```

## 📊 步骤 5: 生成报告

```bash
# 生成 HTML 报告
python generate_report.py --data-dir ../data --output-dir ../docs
```

## 🌐 步骤 6: 查看报告

```bash
# macOS
open ../docs/index.html

# Linux
xdg-open ../docs/index.html

# Windows
start ../docs/index.html
```

## 🎓 下一步

### 本地实验

1. **修改拒答配置**
   ```bash
   # 使用保守模式
   python run_eval.py ... --rejection-preset conservative
   
   # 使用激进模式
   python run_eval.py ... --rejection-preset aggressive
   ```

2. **尝试不同的检索模式**
   ```bash
   # 纯向量检索
   python run_eval.py --retrieval-mode vector ...
   
   # 纯 BM25 检索
   python run_eval.py --retrieval-mode bm25 ...
   ```

3. **扩充测试集**
   ```bash
   # 查看当前统计
   python expand_testset.py --action stats
   
   # 创建新文档
   python expand_testset.py --action create-doc --doc-id 8
   
   # 编辑 corpus/doc-8.txt，添加内容
   # 更新 corpus/queries.json，添加查询
   
   # 验证格式
   python expand_testset.py --action validate
   
   # 重新运行评估
   python run_eval.py ...
   ```

### 部署到 GitHub

4. **配置 GitHub Actions**
   
   a. Push 代码到 GitHub:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/rag-bootcamp.git
   git push -u origin main
   ```
   
   b. 在 GitHub 仓库设置 Secret:
   - 进入 Settings → Secrets and variables → Actions
   - 点击 "New repository secret"
   - Name: `DEEPSEEK_API_KEY`
   - Secret: 粘贴你的 API Key
   - 点击 "Add secret"

5. **启用 GitHub Pages**
   
   - 进入 Settings → Pages
   - Source: "Deploy from a branch"
   - Branch: `main`, Folder: `/reports`
   - 点击 Save
   - 等待 1-2 分钟
   - 访问 `https://YOUR_USERNAME.github.io/rag-bootcamp/`

6. **触发自动评估**
   
   ```bash
   # 添加新文档
   echo "新的文档内容" > iteration7/corpus/doc-8.txt
   
   # 提交并推送
   git add iteration7/corpus/doc-8.txt
   git commit -m "feat: add doc-8"
   git push
   
   # GitHub Actions 自动运行评估
   # 查看 Actions 标签页查看进度
   ```

## 📚 深入学习

- [完整文档](README.md)
- [Iteration 7 详细说明](iteration7/README.md)
- [测试集扩充指南](iteration7/TESTSET_EXPANSION_GUIDE.md)
- [设计文档](docs/iteration-plan.md)

## 🚨 常见问题

### Q: pip install 失败

A: 尝试使用虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Q: ChromaDB 报错

A: 删除旧的数据库：
```bash
rm -rf iteration7/chroma_db
```

### Q: API 调用超时

A: 降低批次大小：
```bash
python run_eval.py ... --batch-size 3
```

### Q: 报告无法打开

A: 检查文件是否存在：
```bash
ls -lh ../docs/index.html
```

## 💡 性能提示

- 使用 `--batch-size 10` 加速（但可能触发限流）
- 使用 `--judge-mode none` 跳过 Judge（测试用）
- 使用 `--no-rejection` 禁用拒答机制（调试用）

## 📞 获取帮助

- 查看 [故障排除](README.md#-故障排除)
- 提交 [GitHub Issue](https://github.com/YOUR_USERNAME/rag-bootcamp/issues)
- 查看 [完整文档](README.md)

---

**准备好了吗？开始你的 RAG 之旅吧！🚀**
