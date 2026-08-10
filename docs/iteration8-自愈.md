# RAG Core 迭代 8 —— 自愈闭环

## 整体设计思路

迭代 7 实现了"每次改动自动评估"的闭环——push 触发 Ragas 评估，产出 Faithfulness / Relevance / Rejection Rate / Hit Rate / MRR 五类指标。但发现问题之后的修复动作仍然是纯人工的：你需要看报告、定位失败 case、手动补 QA 对、再跑一次评估验证。

迭代 8 要解决的问题是：**把"发现 → 分析 → 生成修复方案 → 入库"这条链路的后半段自动化**。系统自己发现被拒答的 query，自己分析原因，自己生成候选 QA 对，你只需要审核确认。这相当于给 RAG 系统装了一个"免疫系统"——自动检测病原、自动生成抗体、人工确认注射。

每一次迭代都遵循同一个模板：

1. **这次迭代解决的是哪个具体问题**
2. **为什么选这个技术方案，而不是别的**
3. **怎么验收**——用什么指标证明这次迭代确实变好了
4. **产出物**——这次迭代结束你手上应该有什么

---

## Iteration 8：自愈闭环

**目的**

把迭代 7 的评估闭环从"只发现问题"升级到"发现问题 + 自动归因 + 自动生成修复候选 + 人工审核入库 + 自动验证效果"。核心不是"AI 自动修一切"，而是"AI 负责繁琐的分析和生成工作，人负责最后的决策确认"。

**具体做什么**

**1. 巡检脚本 (`scripts/daily_review.py`)**

- 跑一遍 Ragas 评估（复用迭代 7 的评估脚本，不重写）
- 遍历所有被拒答的 query，按四层拒答体系分类提取：
  - `query`：原始用户问题
  - `ground_truth_passage`：应该被检索到的正确文档片段
  - `rejection_layer`：哪一层拒答的（layer0/layer1/layer2/layer3）
  - `rejection_reason`：具体原因（`hit==0` / `answer_rank>threshold` / `max_score<阈值` / `coverage<阈值` / `faithfulness<阈值` / `relevance<阈值`）
  - `answer_rank`：正确答案在检索结果中的排名（如有）
- 按优先级排序生成候选：layer0 `hit==0` > layer0 `rank>threshold` > layer1 > layer2 > layer3
- 对每个失败 case，调用 LLM（独立调用，和生成器同模型但不同 session）做两件事：
  - **归因分析**：一句话说清楚为什么拒答
  - **生成候选 QA 对**：基于 ground truth passage 生成 (问, 答) 对，覆盖原始 query 的表述方式
- 输出候选 QA 对到 `review/{date}/candidate_{id}.json`，每条带元数据：

```json
{
  "query": "工厂是几月建造的",
  "answer": "工厂于2023年10月启动建造",
  "source": "ground_truth_passage",
  "trigger": {
    "layer": "layer0",
    "reason": "answer_rank=5, threshold=3",
    "original_rank": 5,
    "evaluation_date": "2026-08-10"
  },
  "original_query": "工厂啥时候建的",
  "analysis": "用户使用了口语化表述'啥时候建的'，与文档中的'建造时间'语义匹配度不够高，导致正确答案排在第5位。建议补充口语化问法的QA对。"
}
```

**2. 审核页面 (`reports/review.html`)**

- 从 `review/` 目录读取所有候选 QA 对列表（通过 GitHub API 或静态 JSON 文件）
- 按日期分组展示，每条显示：
  - 原始 query 和候选 QA 内容
  - 触发层级和原因
  - 归因分析
- 每条有复选框，支持单个勾选、全选、全不选
- "批量入库"按钮 → 收集选中的文件 ID 列表 → 调用 GitHub API 触发 `promote_reviewed` workflow
- 支持"拒绝"操作 → 将选中文件移到 `review/rejected/` 并打标，不删除（保留追溯能力）

**3. 入库 Workflow (`.github/workflows/promote_reviewed.yml`)**

- `workflow_dispatch` 触发，接收网页传来的选中文件列表（JSON 数组）
- 将选中文件从 `review/{date}/` 移到 `docs/qa/`
- 更新 `docs/qa/index.json`（QA 对索引文件，方便检索时加载）
- commit + push → 自动触发迭代 7 的 push 评估 workflow
- 清理 `review/` 中已入库的文件

**4. 清理机制**

- 每次巡检时，检查 `review/` 目录下超过 30 天未被处理的候选 QA 对
- 自动移到 `review/stale/` 并打标，不自动删除（保留人工最终决定权）
- 审核页面显示积压数量提醒

**5. 可追溯性**

- 每次入库操作记录：入库时间、入库文件列表、操作来源（网页 vs 手动）
- 每条入库的 QA 对保留 `trigger` 元数据，作为知识库文档的一部分
- 可回答"这条 QA 对是因为哪个 query 被拒答而生成的"

**为什么这么选**

- **自动生成候选，不直接入库**：LLM 生成的 QA 对可能措辞不够好、或者和已有文档重复。人工审核确保只入库真正有用的内容，同时审核成本很低——每条候选看 5 秒决定通过/拒绝，远比从零写 QA 对快
- **基于 ground truth 生成，不是凭空编造**：候选 QA 对的答案来源于已有文档的 ground truth passage，LLM 只负责把答案和原始 query 对齐成 QA 格式。这避免了"用幻觉修复幻觉"的循环
- **物理隔离 review/ 和 docs/**：候选 QA 对不会污染正式知识库，审核不通过的直接拒绝或归档，不影响线上检索
- **按日期分组**：方便追溯"这批候选是哪个时间点产生的"，也方便定期清理积压
- **复用 GitHub Pages**：审核页面和评估报告放在同一个站点，你不需要额外部署服务。GitHub Actions 负责所有后端操作
- **四层拒答全部触发自愈，不只 layer0**：检索没命中要补、排名不够要补、引用幻觉多要补、faithfulness 低也可能要补——每一层暴露的问题都值得生成候选，只是优先级不同

**验收标准**

- 手动跑一次 `daily_review.py`，确认：
  - 所有被拒答的 query 都生成了候选 QA 对
  - 候选 QA 对包含完整的 `trigger` 元数据和归因分析
  - 候选文件写入 `review/{date}/` 目录，不写入 `docs/`
- 打开 GitHub Pages 审核页面，确认：
  - 能看到所有待审核候选，按日期分组
  - 能勾选部分候选并点击"批量入库"
  - 入库后文件从 `review/` 消失，出现在 `docs/qa/`
  - 能拒绝候选，文件移到 `review/rejected/`
- 入库后 push 自动触发评估，下次评估报告中：
  - 原本被拒答的 query 现在能命中新入库的 QA 对
  - Hit Rate 提升，Rejection Rate 下降（具体数值取决于候选 QA 对数量和质量）
- 审核页面显示超过 30 天未处理的候选数量

**产出物**

- `scripts/daily_review.py`：巡检脚本（评估 + 归因 + 候选 QA 对生成）
- `reports/review.html`：审核页面
- `.github/workflows/daily_review.yml`：每日定时巡检 workflow（`schedule` + `workflow_dispatch`）
- `.github/workflows/promote_reviewed.yml`：入库操作 workflow
- `review/` 目录结构 + `docs/qa/` 目录结构 + `docs/qa/index.json` 索引文件
- 迭代 8 运行一周后的效果报告（Hit Rate 变化、Rejection Rate 变化、审核效率统计）

---

## 和已有系统的衔接

| 已有组件 | 迭代 8 怎么复用 |
|---------|---------------|
| Ragas 评估脚本（迭代 7） | `daily_review.py` 直接调用，不重写 |
| 四层拒答配置（迭代 6） | 读取已有 `rejection_config.json`，每层的阈值就是触发条件 |
| GitHub Pages 评估报告（迭代 7） | 新增一个 review tab 或独立页面，不是重做整个站点 |
| push 触发评估 workflow（迭代 7） | 入库后的 push 自动触发，不新增 workflow，完全复用 |
| ground truth passage（迭代 0） | 候选 QA 对的答案来源，LLM 不编造内容 |

---

## 自愈闭环完整链路

```
每日定时巡检 (GitHub Actions schedule / workflow_dispatch)
    ↓
跑 Ragas 评估 + 四层拒答
    ↓
收集所有被拒答的 query，按优先级排序
    ↓
LLM 独立调用：归因分析 + 基于 ground truth 生成候选 QA 对
    ↓
存入 review/{date}/ 目录 → commit + push
    ↓
GitHub Pages 审核页面自动更新
    ↓
你打开网页 → 审核 → 勾选通过的，拒绝无效的 → 批量入库
    ↓
workflow_dispatch 将选中文件移到 docs/qa/，更新索引
    ↓
push 触发迭代 7 评估 workflow → 验证效果
    ↓
（循环）新的评估可能暴露新的拒答 → 继续自愈
```

---

## 迭代 8 在整个 RAG 计划中的位置

迭代 7 之后，自愈闭环是让整个 RAG 系统从"可评估"升级到"可自愈"的最后一步。它不引入新的检索算法或生成策略，而是在已有的评估体系和拒答机制之上，用工程手段把修复效率从"小时级人工操作"降到"分钟级审核确认"。这一步做完，你的 RAG 系统就是一个完整的、可持续进化的知识问答引擎。