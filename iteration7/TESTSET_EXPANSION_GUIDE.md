# 测试集扩充指南

## 📋 目标

将测试集从当前规模扩充到生产级别：
- **文档数量**：7 个 → 20-30 个
- **查询数量**：30-50 条 → 100-300 条
- **覆盖率**：确保每个文档至少有 3-5 个查询

## 🎯 扩充策略

### 1. 从误判案例中提取

**来源**：Iteration 6 的分析结果

查看 `iteration6/results.json`，找出：
- **Hit = 0** 的查询（检索失败）
- **Faithfulness < 0.5** 的查询（生成质量差）
- **被拒答**的查询（但实际应该回答）

**操作**：
```bash
# 分析现有结果，找出问题案例
cd iteration6
python -c "
import json
with open('results.json') as f:
    data = json.load(f)
    for r in data['results']:
        if r['hit'] == 0:
            print(f\"Miss: Q{r['id']} - {r['query']}\")
"
```

### 2. 增加边界情况

| 类型 | 说明 | 示例 |
|------|------|------|
| **极短文档** | < 100 字 | 简短的产品规格、快速提示 |
| **超长文档** | > 5000 字 | 详细的技术手册、完整教程 |
| **多主题文档** | 包含多个独立话题 | FAQ 集合、多功能说明 |
| **模糊文档** | 信息不完整或有歧义 | 测试拒答机制的准确性 |

### 3. 增加查询难度梯度

| 难度 | 类型 | 特征 | 占比建议 |
|------|------|------|---------|
| **简单** | factual | 直接提取，单一事实 | 40% |
| **中等** | reasoning | 需要推理，综合信息 | 40% |
| **困难** | multi-hop | 跨段落/跨文档，多步推理 | 20% |

## 🛠️ 使用扩充工具

### 查看当前统计

```bash
cd iteration7
python expand_testset.py --action stats
```

输出示例：
```
📊 Corpus & Queries Statistics
================================================

📁 Documents: 7
   Total chars: 15,234
   Average: 2,176 chars/doc

Documents:
   • doc-1.txt: 2,345 chars, 45 lines
   • doc-2.txt: 1,890 chars, 38 lines
   ...

❓ Queries: 35
   By Category:
      • factual: 15
      • reasoning: 12
      • multi-hop: 8
   
   By Document:
      • doc-1: 6 queries
      • doc-2: 4 queries
      ...
   
   Coverage:
      • Documents covered: 7/7 (100%)
```

### 验证格式

```bash
python expand_testset.py --action validate
```

输出示例：
```
🔍 Validation Results
================================================

📁 Corpus Validation:
   ✅ All documents are valid

❓ Queries Validation:
   ❌ Query 12: Document not found (doc-8.txt)
   ⚠️  Query 15: char_start >= char_end
```

### 创建新文档

```bash
# 创建 doc-8.txt 模板
python expand_testset.py --action create-doc --doc-id 8
```

会生成 `corpus/doc-8.txt`，包含：
```
# Document 8

[在这里填写文档内容]

提示：
1. 文档内容应该包含可以回答问题的信息
2. 建议字数：200-1000 字
...
```

### 创建新查询

```bash
# 生成查询模板
python expand_testset.py --action create-query
```

输出模板：
```json
{
  "id": 36,
  "query": "[在这里填写查询问题]",
  "doc_id": "X",
  "char_start": 0,
  "char_end": 100,
  "category": "factual|reasoning|multi-hop"
}
```

## 📝 扩充工作流

### Step 1: 规划

确定要添加的文档主题：

```bash
# 查看当前覆盖的主题
cd iteration7
ls -1 corpus/doc-*.txt

# 规划新主题（示例）
# doc-8.txt: 退货政策
# doc-9.txt: 物流信息
# doc-10.txt: 会员权益
```

### Step 2: 添加文档

```bash
# 创建模板
python expand_testset.py --action create-doc --doc-id 8

# 编辑 corpus/doc-8.txt，填入实际内容
```

### Step 3: 添加查询

对于每个新文档，添加 3-5 个查询：

1. 生成查询模板：
```bash
python expand_testset.py --action create-query
```

2. 手动编辑 `corpus/queries.json`，添加查询：
```json
{
  "id": 36,
  "query": "退货需要哪些条件？",
  "doc_id": "8",
  "char_start": 150,
  "char_end": 280,
  "category": "factual"
}
```

3. 定位答案位置（`char_start` 和 `char_end`）：
```python
# 辅助脚本：找到答案在文档中的位置
with open('corpus/doc-8.txt') as f:
    content = f.read()
    answer = "需要在购买后7天内，保持商品完好"
    start = content.find(answer)
    end = start + len(answer)
    print(f"char_start: {start}, char_end: {end}")
```

### Step 4: 验证

```bash
# 验证新增的文档和查询
python expand_testset.py --action validate

# 查看更新后的统计
python expand_testset.py --action stats
```

### Step 5: 测试评估

```bash
# 运行评估，检查新文档/查询的表现
python run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode hybrid \
  --rerank-mode bge \
  --output-dir ../data
```

### Step 6: 提交代码

```bash
# 提交到 Git，触发自动评估
git add corpus/
git commit -m "feat: add doc-8 and related queries"
git push origin main
```

## 📊 扩充进度追踪

创建一个 checklist：

```markdown
## 文档扩充进度

- [x] doc-1 ~ doc-7 (已有)
- [ ] doc-8: 退货政策 (3 queries)
- [ ] doc-9: 物流信息 (4 queries)
- [ ] doc-10: 会员权益 (3 queries)
- [ ] doc-11: 支付方式 (3 queries)
- [ ] doc-12: 账户安全 (4 queries)
...

目标：20-30 个文档，100-300 条查询
当前：7 个文档，35 条查询
完成度：35% (文档), 35% (查询)
```

## 🎯 质量检查清单

添加每个新文档/查询后，确认：

### 文档质量
- [ ] 内容完整，没有明显错误
- [ ] 长度适中（200-1000 字）
- [ ] 包含可以回答问题的信息
- [ ] 格式清晰（分段、标点）

### 查询质量
- [ ] 问题清晰、具体
- [ ] 答案确实在指定文档中
- [ ] `char_start` 和 `char_end` 准确
- [ ] `category` 分类正确
- [ ] 难度分布合理（简单/中等/困难）

### 覆盖率
- [ ] 每个文档至少有 3 个查询
- [ ] 三种 category 都有覆盖
- [ ] 包含正面/负面/边界情况

## 💡 最佳实践

### 1. 增量扩充

不要一次性添加所有文档，而是：
- 每次添加 2-3 个文档
- 运行评估，观察指标变化
- 根据结果调整后续扩充策略

### 2. 多样性优先

确保文档和查询的多样性：
- 不同长度的文档
- 不同难度的查询
- 不同领域的知识

### 3. 真实案例优先

从实际业务场景中收集：
- 用户常问的问题
- 客服记录
- 产品反馈

### 4. 定期清理

扩充过程中发现质量差的查询，及时删除或修改。

## 🚨 常见问题

### Q1: 如何快速定位 char_start 和 char_end？

```python
# 辅助脚本
def find_answer_position(doc_file, answer_text):
    with open(doc_file, 'r', encoding='utf-8') as f:
        content = f.read()
        start = content.find(answer_text)
        if start == -1:
            print(f"❌ Answer not found in document")
            return None
        end = start + len(answer_text)
        print(f"✅ Found at char_start={start}, char_end={end}")
        print(f"Context: ...{content[max(0, start-20):end+20]}...")
        return (start, end)

# 使用
find_answer_position('corpus/doc-8.txt', '购买后7天内')
```

### Q2: 如何确保查询的答案覆盖正确？

运行一次评估，检查 `hit` 指标：
```bash
python run_eval.py --retrieval-mode hybrid --rerank-mode bge
# 查看 results.json，检查新查询的 hit=1
```

### Q3: 扩充后指标下降怎么办？

这是正常的！原因：
1. 新增的查询可能更难
2. 新文档可能包含更多噪声

应对策略：
- 分析具体哪些查询失败
- 调整 chunking/retrieval 参数
- 优化拒答阈值

### Q4: 如何平衡扩充速度和质量？

建议节奏：
- **Week 1**: 添加 5 个文档，15-20 个查询
- **Week 2**: 添加 5 个文档，20-25 个查询
- **Week 3**: 添加 5 个文档，25-30 个查询
- **Week 4**: 质量审查，清理低质量查询

## 📚 参考资源

- [Iteration 6 分析结果](../iteration6/results.json)
- [拒答配置](rejection_config.json)
- [评估脚本](run_eval.py)

---

**Happy Expanding! 🚀**
