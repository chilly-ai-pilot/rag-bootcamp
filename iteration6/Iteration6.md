# Iteration 6: 引用验证机制 (Citation Validation)

## 背景

在 Iteration 5 中，我们引入了 LLM-as-Judge 评估系统，发现了一些低分案例（Faithfulness < 0.8）：

| Query | 问题 | Faithfulness | 根本原因 |
|-------|------|--------------|----------|
| Q5 | ST-500 安装高度 | 0.65 | 引用标签错误（[文档13:片段2] vs [文档13:片段1]） |
| Q22 | 防止摄像头被偷看 | 0.75 | 跨文档混淆 + 缺失前置条件 |
| Q23 | 插座防过充 | 0.75 | 过度泛化（USB → 整个插座） |

**核心问题**：LLM 会"编造"引用标签，或将错误的标签标注到内容上。

## 解决方案：Span-Based Citation + 三层验证

### 1. 输出格式改进

**旧版本** (Iteration 5):
```
答案：ST-500安装高度为1.8-2.2米[文档13:片段2]...
问题：标注和内容混在一起，难以验证
```

**新版本** (Iteration 6):
```json
{
  "answer": "ST-500安装高度为1.8-2.2米，步骤是先按压粘贴再连接电源。",
  "citations": [
    {"span": "安装高度为1.8-2.2米", "source": "文档13:片段1"},
    {"span": "先按压粘贴再连接电源", "source": "文档13:片段2"}
  ]
}
```

**优势**：
- answer 和 citations 分离
- span 必须是 answer 的完整子串（可验证）
- 使用 `response_format={"type": "json_object"}` 强制 JSON 输出

### 2. Prompt 改进：明确列出可用标签

**旧版本**：没有列出可用标签，LLM 会瞎编
```
回答时用[文档X:片段N]标注引用来源...
```

**新版本**：明确列出所有合法标签
```
**可用的引用标签（请只使用这些标签，不要编造）**：
[文档13:片段1], [文档13:片段2], [文档5:片段3], [文档5:片段4]
```

### 3. 三层验证机制

#### 层级 1: Span 存在性验证
```python
if span not in answer:
    reason = "span不在answer中"
```
- 确保 span 是 answer 的完整子串
- 追踪位置占用，避免重复标注

#### 层级 2: Source 合法性验证
```python
if source not in valid_sources:
    reason = "source不存在于合法来源列表"
```
- 拒绝 LLM 编造的标签（如 [文档13:片段999]）

#### 层级 3: 内容一致性验证

使用**梯度验证策略**：

**3.1 精确匹配**（最可靠）
```python
if span in source_text:
    return True  # 精确匹配，最可靠
```

**3.2 数字上下文验证**（防止张冠李戴）
```python
numbers_match_context(span, source_text, window=15)
```

关键技术：
- 提取 span 中的所有数字（如 1.8, 2.2, 36）
- 检查数字在 source 中的上下文（前后 15 字符）
- 要求上下文与 span 有关键词重叠

示例：
```python
# 错误：数字对不上上下文
span = "工作电压36V"
source = "ST-500工作电压24V"  # 24V != 36V
→ 验证失败

# 正确：数字上下文匹配
span = "安装高度1.8米"
source = "安装高度1.8-2.2米"  # "安装高度" 在数字周围
→ 验证通过
```

**3.3 词汇重叠度验证**
```python
coverage = len(span_terms & source_terms) / len(span_terms)
return coverage >= 0.5
```

关键技术：**2-gram 滑窗**（来自 Claude 测试代码启发）
```python
# 避免中文分词边界不一致
chinese_only = re.sub(r'[^\u4e00-\u9fa5]', '', text)
bigrams = {chinese_only[i:i+2] for i in range(len(chinese_only) - 1)}

# 示例："安装高度" → {"安装", "装高", "高度"}
```

### 4. 渲染策略：从后往前插入

**问题**：从前往后插入会导致位置偏移
```python
answer = "1234567890"
# 插入第一个标注后
answer = "12[tag1]34567890"  
# 第二个标注的位置已经偏移了！
```

**解决**：从后往前插入
```python
for item in sorted(passed, key=lambda x: x["pos"], reverse=True):
    insert_pos = item["pos"] + len(item["span"])
    label = f"[{item['source']}]"
    final_answer = final_answer[:insert_pos] + label + final_answer[insert_pos:]
```

### 5. 失败处理策略

**测试阶段**：记录所有失败案例到 JSON
```json
{
  "validation": {
    "failed": [
      {
        "span": "工作电压为36V",
        "source": "文档13:片段1",
        "reason": "span内容在source原文中找不到充分依据（数字上下文不匹配）"
      }
    ],
    "stats": {
      "total": 3,
      "passed": 2,
      "failed": 1,
      "pass_rate": 0.667
    }
  }
}
```

**后续策略**（根据失败率决定）：
- 失败率低 (<5%)：删除失败的 citations
- 失败率中 (5-20%)：保留但添加警告标记
- 失败率高 (>20%)：警告用户或重新生成

## 代码结构

### citation_validator.py

核心函数：
```python
def extract_terms(text: str) -> Set[str]:
    """提取 2-gram + 型号 + 数字"""
    
def numbers_match_context(span: str, source_text: str, window: int = 15) -> bool:
    """验证数字上下文"""
    
def span_supported_by_source(span: str, source_text: str, threshold: float = 0.5) -> bool:
    """三层梯度验证"""
    
def validate_and_render(answer: str, citations: List[Dict], valid_sources: Dict) -> Dict:
    """主函数：验证 + 渲染"""
```

### generation.py (V6)

新增函数：
```python
def _build_context_v6(retrieved_chunks):
    """构建上下文 + 提取所有合法标签"""
    return context_str, valid_labels, chunks_map

def generate_answer_v6(query: str, retrieved_chunks: list, ...) -> Dict:
    """带引用验证的生成函数"""
    return {
        "answer": "最终答案（带标注）",
        "raw_answer": "原始答案（不含标注）",
        "citations": [...],
        "validation": {
            "enabled": bool,
            "passed": [...],
            "failed": [...],
            "stats": {...}
        },
        "llm_raw_response": "..."
    }

async def generate_answer_v6_async(...):
    """异步版本（用于批量评估）"""
```

### run_eval.py 集成

新增参数：
```bash
--generation-version v6  # 使用新版本
--validation-threshold 0.5  # 词汇重叠度阈值
```

新增输出：
```
=== Citation Validation Statistics (Iteration 6) ===
  Total queries evaluated: 32
  Total citations:         96
  Passed:                  89 (92.7%)
  Failed:                  7 (7.3%)
  
  Failed citations by reason:
    • span内容在source原文中找不到充分依据: 5
    • source不存在于合法来源列表: 2
```

## 预期改进

基于 Iteration 5 的低分案例：

| Query | 问题类型 | Iteration 5 | 预期 Iteration 6 | 改进机制 |
|-------|---------|-------------|------------------|----------|
| Q5 | 标签错误 | 0.65 | 0.85+ | 拒绝不存在的标签 |
| Q22 | 跨文档混淆 | 0.75 | 0.80+ | 分别标注每个来源 |
| Q23 | 过度泛化 | 0.75 | 0.80+ | 精确匹配原文内容 |

**总体目标**：
- Faithfulness 平均分：0.938 → 0.95+
- 引用幻觉率：~10% → <5%
- 验证通过率：期望 >90%

## 技术亮点

### 1. Claude 测试代码启发

从用户提供的 Claude 测试代码中学到的关键技术：

**2-gram 滑窗**：
```python
# 为什么不用 jieba 分词？
# 原因：分词边界不一致会导致误判
# 示例："安装高度" vs "安" "装" "高度"
chinese_only = re.sub(r'[^\u4e00-\u9fa5]', '', text)
bigrams = {chinese_only[i:i+2] for i in range(len(chinese_only) - 1)}
```

**数字上下文窗口**：
```python
# 防止"张冠李戴"：不同数字恰好相同
for m in re.finditer(re.escape(num), source_text):
    ctx = source_text[max(0, m.start() - window): m.end() + window]
    if extract_terms(ctx) & span_terms:
        ok = True
```

**位置占用追踪**：
```python
# 避免多个 citation 定位到同一段文字
if not any(s <= idx < e for s, e in occupied):
    pos = idx
occupied.append((pos, pos + len(span)))
```

### 2. Prompt Engineering

**关键改进**：明确列出所有可用标签
```
**可用的引用标签（请只使用这些标签，不要编造）**：
[文档13:片段1], [文档13:片段2], [文档5:片段3]
```

这比"请用[文档X:片段N]格式"要明确得多！

### 3. 结构化输出 + 强制 JSON 模式

```python
resp = client.chat.completions.create(
    model=model,
    response_format={"type": "json_object"},  # 强制 JSON
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT_V6},
        {"role": "user", "content": prompt}
    ]
)
```

## 使用方法

### 1. 测试引用验证器

```bash
cd iteration6
python3 citation_validator.py
```

预期输出：
```
==========================================================
Citation Validator 测试
==========================================================

原始答案:
ST-500人体传感器安装高度为1.8-2.2米，安装步骤是先按压粘贴，再连接电源完成配网。工作电压为36V。

原始引用:
  1. '安装高度为1.8-2.2米' → 文档13:片段1
  2. '先按压粘贴，再连接电源完成配网' → 文档13:片段2
  3. '工作电压为36V' → 文档13:片段1

==========================================================
验证结果
==========================================================

✅ 通过验证: 2/3
❌ 未通过: 1/3
📊 通过率: 66.7%

最终答案（带标注）:
ST-500人体传感器安装高度为1.8-2.2米[文档13:片段1]，安装步骤是先按压粘贴，再连接电源完成配网[文档13:片段2]。工作电压为36V。

未通过验证的引用:
  ❌ '工作电压为36V' → 文档13:片段1
     原因: span内容在source原文中找不到充分依据（词汇重叠度不足或数字上下文不匹配）
```

### 2. 运行完整评估（V6 版本）

```bash
# 使用新版本生成函数（带验证）
python3 run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode hybrid \
  --rerank-mode bge \
  --judge-mode llm \
  --generation-version v6 \
  --validation-threshold 0.5
```

### 3. 分析验证失败率

```python
import json

with open('results_fixed_100_50_hybrid.json') as f:
    data = json.load(f)

# 提取验证统计
validation = data.get('validation_analysis', {})
print(f"引用验证通过率: {validation.get('pass_rate', 0):.1%}")
print(f"失败的引用数: {validation.get('failed_citations', 0)}")

# 查看失败案例
for fail in validation.get('failed_details', [])[:5]:
    print(f"\nQuery {fail['query_id']}: {fail['query']}")
    print(f"  Span: '{fail['span']}'")
    print(f"  Source: {fail['source']}")
    print(f"  Reason: {fail['reason']}")
```

## 成本分析

**无额外 API 成本**：
- 验证在本地完成，不调用 LLM
- JSON 模式不增加 token 消耗
- 与 Iteration 5 成本相同

**额外计算成本**：
- 2-gram 提取：O(n)
- 数字上下文验证：O(m*k)，m 为数字数量，k 为出现次数
- 总体验证时间：<10ms per citation

## 下一步

1. ✅ 实现 citation_validator.py
2. ✅ 更新 generation.py（V6 版本）
3. ✅ 更新 run_eval.py（集成验证）
4. ⏳ 运行完整评估（32 queries）
5. ⏳ 分析验证失败率和失败原因分布
6. ⏳ 对比 Iteration 5 vs Iteration 6 的 Faithfulness 分数
7. ⏳ 根据失败率决定处理策略
8. ⏳ 如果验证失败率高，可能需要调整 threshold 或改进 prompt

## 参考资料

- Iteration 5 评估报告: `iteration5/judge_analysis_report.md`
- Iteration 5 结果文件: `iteration5/results_fixed_100_50_vector.json`
- Claude 测试代码: 用户在对话历史中提供
- 低分案例: Query 5 (0.65), Query 22 (0.75), Query 23 (0.75)
