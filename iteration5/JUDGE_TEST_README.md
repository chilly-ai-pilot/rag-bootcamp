# Judge 测试指南

本文档说明如何测试和比较 LLM Judge 和 Ragas Judge 的性能表现。

---

## 📋 目录

- [测试目标](#测试目标)
- [测试数据集](#测试数据集)
- [快速开始](#快速开始)
- [测试结果对比](#测试结果对比)
- [使用建议](#使用建议)

---

## 🎯 测试目标

验证两个 Judge 系统对错误答案的识别能力：

1. **LLM Judge** - 使用阿里云 Qwen 模型，通过详细的逐句分析评估答案忠实度
2. **Ragas Judge** - 使用 Ragas 框架，通过自动化指标评估答案质量

测试维度：
- ✅ **错误答案识别率**：能否识别出故意制造的错误答案
- ✅ **正确答案误判率**：是否会把正确答案误判为错误
- ✅ **不同错误类型的识别能力**：对不同难度错误的识别表现

---

## 📚 测试数据集

### 错误答案测试集（20条）

测试集包含三种类型的故意错误，难度递增：

#### **Type A: Entity Replacement (实体替换)** - 8条 - Easy

修改了关键实体或数值，其他内容不变。

**样例**：
```
问题：SmartLock-100 使用什么电池？
正确：SmartLock-100 使用 4 节 5 号电池，续航约 1 年。
错误：SmartLock-100 使用 4 节 7 号电池，续航约 1 年。
       ↑ 只改了一个数字
```

**其他样例**：
- 120° → 180°（角度错误）
- 10米 → 15米（距离错误）
- 4节 → 8节（数量错误）

---

#### **Type B: Causal Error (因果错误)** - 6条 - Medium

破坏了因果关系、条件逻辑或操作步骤。

**样例**：
```
问题：SmartPlug-400 的充电保护功能是如何工作的？
正确：检测到设备电流在短时间内下降至接近零，判定充满电
错误：检测到设备电流在短时间内上升至峰值，判定充满电
       ↑ 因果关系错误：充满电时电流应该下降，不是上升
```

**其他样例**：
- 低于15% → 高于15%（条件矛盾）
- 扣板松动 → 扣板紧固（原因错误）
- 先按下再下压 → 先按下后松开（步骤错误）

---

#### **Type C: Cross-Document (跨文档混淆)** - 6条 - Hard

把A产品的信息错误地应用到B产品。

**样例**：
```
问题：SmartLock-100 的电池续航多久？
正确：SmartLock-100 使用 4 节 5 号电池，续航约 1 年。
错误：SmartLock-100 使用 4 节 5 号电池，续航约 2 年。
       ↑ "2年"是 ST-500 的续航，被错误地应用到 SmartLock-100
```

**其他样例**：
- ST-500 的电池型号应用到其他传感器
- GW-200 的功能应用到 SmartLock-100
- ST-501 的精度应用到 ST-500

---

### 正确答案对照组（10条）

从 `queries.json` 中选取的真实正确答案，用于测试误判率。

---

## 🚀 快速开始

### 1. 环境准备

确保已配置必要的 API 密钥：

```bash
# 阿里云 Qwen API (用于 LLM Judge)
export ALI_API_KEY="your-ali-api-key"
export ALI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# DeepSeek API (用于生成答案，可选)
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

### 2. 测试 LLM Judge

```bash
cd iteration5
python3 test_judge_llm.py
```

**预计时间**：2-3分钟（异步批量处理，batch_size=10）

**输出文件**：`judge_test_results_detailed.json`

### 3. 测试 Ragas Judge

```bash
cd iteration5
python3 test_judge_ragas.py
```

**预计时间**：3-5分钟（Ragas 评估较慢）

**输出文件**：`judge_test_results_ragas.json`

---

## 📊 测试结果对比

### 完整对比表

| 维度 | LLM Judge | Ragas Judge | 胜者 |
|------|-----------|-------------|------|
| **错误答案总识别率** | **100%** (20/20) | 85% (17/20) | 🏆 LLM |
| **Type A 实体替换** | **100%** (8/8) | **100%** (8/8) | 🤝 平局 |
| **Type B 因果错误** | **100%** (6/6) | **100%** (6/6) | 🤝 平局 |
| **Type C 跨片段拼接** | **100%** (6/6) | 50% (3/6) | 🏆 LLM |
| **正确答案误判率** | 0% (0/10) | 0% (0/10) | 🤝 平局 |
| **错误答案平均分数** | **0.17** (更低更好) | 0.20 | 🏆 LLM |
| **正确答案平均分数** | 1.00 | 1.00 | 🤝 平局 |
| **提供审核理由** | ✅ 详细逐句分析 | ❌ 只有分数 | 🏆 LLM |
| **评估速度** | 较慢 (2-3分钟) | 较慢 (3-5分钟) | 🤝 相近 |

---

### 关键发现

#### 1. **LLM Judge 全面领先** 🏆

**优势**：
- ✅ 识别率更高：100% vs 85%
- ✅ 对复杂错误（Type C）识别完美：100% vs 50%
- ✅ 给分更严格：0.17 vs 0.20
- ✅ 提供详细理由，便于人工复核
- ✅ 零误判率

**劣势**：
- ⚠️ 依赖 LLM API（成本、延迟）
- ⚠️ 需要精心设计 prompt

---

#### 2. **Ragas Judge 的局限性** ⚠️

**优势**：
- ✅ Type A/B 识别能力尚可（100%）
- ✅ 零误判率
- ✅ 开源框架，易于集成

**劣势**：
- ❌ **Type C (跨片段拼接) 识别能力弱** - 只有 50%
- ❌ **3个查询返回 NaN** - 稳定性问题
- ❌ **无法提供审核理由** - 不便于调试
- ❌ 评估速度并不快

---

#### 3. **分数分布对比**

**LLM Judge**：
```
错误答案：
  - 分数范围：[0.00, 0.67]
  - 平均分数：0.17
  - 分布：11个低分(<0.3), 6个中分(0.3-0.7), 0个高分(≥0.7)

正确答案：
  - 分数范围：[1.00, 1.00]  
  - 平均分数：1.00
  - 所有正确答案都得满分
```

**Ragas Judge**：
```
错误答案：
  - 分数范围：[0.00, 0.67]
  - 平均分数：0.20
  - 有效结果：17/20 (3个返回NaN)

正确答案：
  - 分数范围：[1.00, 1.00]
  - 平均分数：1.00
  - 有效结果：8/10 (2个返回NaN)
```

**结论**：LLM Judge 分数区分度更好，更稳定。

---

## 💡 使用建议

### 1. 生产环境推荐方案

#### **方案A：纯 LLM Judge**（推荐）

适用于：对质量要求高的场景

```
优点：
  ✅ 识别能力最强（100%）
  ✅ 提供详细理由，便于人工复核
  ✅ 稳定性好，无 NaN 问题
  
缺点：
  ⚠️ API 调用成本
  ⚠️ 需要精心设计 prompt
```

---

#### **方案B：两级过滤**（性价比最高）

适用于：需要平衡成本和质量的场景

```
第一道防线：Ragas Judge (快速筛选)
  - Ragas 分数 ≥ 0.5 → 直接通过
  
第二道防线：LLM Judge (详细审核)
  - Ragas 分数 < 0.5 → 交给 LLM Judge 详细审核
  
优点：
  ✅ 大部分正确答案快速通过（节省成本）
  ✅ 可疑答案详细审核（保证质量）
  ✅ 结合两者优势
```

---

#### **方案C：纯 Ragas Judge**（不推荐）

适用于：只能用开源方案的场景

```
缺点：
  ❌ Type C 识别能力弱（50%）
  ❌ 有 NaN 稳定性问题
  ❌ 无法提供审核理由
  
如果必须使用：
  ⚠️ 需要人工复核 Type C 类型的答案
  ⚠️ 需要处理 NaN 情况（如使用默认分数 0.5）
```

---

### 2. Faithfulness 阈值设置

基于测试结果，建议阈值：

| Judge 类型 | 推荐阈值 | 理由 |
|-----------|---------|------|
| **LLM Judge** | **0.5** | 错误答案平均0.17，正确答案1.00，0.5是良好分界点 |
| **Ragas Judge** | **0.5-0.6** | 错误答案平均0.20，正确答案1.00，稍高阈值更安全 |

**阈值含义**：
- 分数 < 阈值 → 答案不可信，触发警告或拒答
- 分数 ≥ 阈值 → 答案可信，正常返回

---

### 3. 特殊场景建议

#### **对产品参数敏感的场景**（如：电商、技术文档）

推荐：**LLM Judge**

理由：Type A (实体替换) 识别完美，能准确捕捉数值、型号错误

---

#### **需要理解因果逻辑的场景**（如：故障排查、操作指南）

推荐：**LLM Judge**

理由：Type B (因果错误) 识别完美，能理解逻辑关系

---

#### **多产品混合文档库**（如：产品手册集合）

推荐：**LLM Judge + 人工复核**

理由：Type C (跨文档混淆) 最难识别，LLM Judge 100%但 Ragas 只有50%

---

## 📁 测试结果文件说明

### LLM Judge 结果文件

**文件名**：`judge_test_results_detailed.json`

**关键字段**：
```
false_claims_results:
  - entity_replacement: Type A 结果
  - causal_error: Type B 结果  
  - cross_document: Type C 结果
  - 每种类型包含：total, detected, accuracy, avg_score, details

correct_answers_results:
  - total, false_positive, false_positive_rate, avg_score

summary:
  - 各类型识别率汇总
```

---

### Ragas Judge 结果文件

**文件名**：`judge_test_results_ragas.json`

**关键字段**：
```
false_claims: 错误答案测试结果列表
correct_answers: 正确答案测试结果列表
threshold: 使用的阈值（0.7）
```

---

## ⚠️ 已知问题

### Ragas Judge 的 NaN 问题

**现象**：约 17% 的查询返回 NaN 分数

**原因**：
- Ragas 内部计算异常
- 输入文本格式问题
- 文档长度极端情况

**临时解决方案**：
```python
if math.isnan(score):
    score = 0.5  # 使用中性分数
```

**长期方案**：
- 升级 Ragas 版本
- 或切换到 LLM Judge

---

## 🔧 故障排查

### 问题1：API 连接失败

```
❌ Judge LLM 连接失败: Connection error
```

**解决方案**：
```bash
# 检查环境变量
echo $ALI_API_KEY
echo $ALI_BASE_URL

# 重新设置
export ALI_API_KEY="your-key"
export ALI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

---

### 问题2：文档读取失败

```
⚠️ 无法提取文档: doc-1.txt
```

**解决方案**：
```bash
# 检查当前目录
pwd  # 应该在 iteration5 目录

# 检查文档
ls -la corpus/doc-*.txt
```

---

### 问题3：Ragas 评估卡住

```
Evaluating:   0%|          | 0/1 [00:00<?, ?it/s]
(长时间无响应)
```

**解决方案**：
- 检查网络连接（Ragas 需要访问 HuggingFace）
- 检查是否需要代理
- 或使用 LLM Judge 替代

---

## 📈 下一步

完成测试后，你可以：

1. **集成到 run_eval.py**
   - 使用测试结果确定的最优 Judge
   - 设置合适的 faithfulness 阈值

2. **实现 Iteration 6 的拒答机制**
   - 当 faithfulness_score < 阈值时，拒绝返回答案
   - 或提示用户"答案可信度较低"

3. **持续监控**
   - 定期运行测试，验证 Judge 性能
   - 收集生产环境中的误判案例，改进 Judge

---

## 📚 参考资料

- [Ragas 官方文档](https://docs.ragas.io/)
- [LLM-as-Judge 论文](https://arxiv.org/abs/2306.05685)
- [Faithfulness 评估最佳实践](https://docs.ragas.io/en/stable/concepts/metrics/faithfulness.html)

---

**更新日期**: 2026-08-06  
**版本**: v2.0
