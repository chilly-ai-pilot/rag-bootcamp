# Iteration 7: 持续评估闭环

## 目的
把Iteration 5的自动化评估，从"手动跑一次"变成"每次改动自动跑"，形成"用真实案例反过来修正阈值"的闭环。这是把整个项目从"能跑的demo"升级到"可维护的系统"的最后一步。

## 技术选型

| 组件 | 选择 | 为什么 |
|---|---|---|
| 自动化触发 | GitHub Actions | 轻量级CI/CD方案，无需额外基础设施 |
| 结果存储 | JSON文件 + Git版本控制 | 简单可追溯，适合学习项目规模 |
| 可视化报告 | 静态HTML + Chart.js | 通过GitHub Pages直接访问，无需后端服务 |
| 测试集扩充 | 逐步从7个文档扩充到20-30个 | 加入误判的真实案例，让测试集随系统迭代变准 |

## 架构设计

### 触发流程
```
corpus/doc-x.txt 新增
    ↓
GitHub Actions 触发
    ↓
运行 run_eval.py
    ↓
保存结果到 data/results_TIMESTAMP.json
    ↓
运行 generate_report.py
    ↓
生成 reports/index.html
    ↓
提交回仓库 + 部署到 GitHub Pages
```

### 目录结构
```
/data/                          # 评估结果历史记录
  - baseline.json               # 基线结果（第一次运行）
  - results_20260807_143022.json
  - results_20260808_091534.json
  
/reports/                       # HTML报告
  - index.html                  # 主报告页面
  - assets/                     # CSS/JS资源
  
/iteration7/                    # 本次迭代代码
  - run_eval.py                 # 增强版评估脚本
  - generate_report.py          # 报告生成器
  - corpus/                     # 扩充的测试集
  - ...
```

## 核心功能

### 1. 增强的评估脚本
- 接受命令行参数指定输出路径
- 生成带时间戳的结果文件
- 包含元数据：评估时间、文档数量、模型配置

### 2. HTML报告生成器
- **趋势图**：Faithfulness、Recall@K 随时间变化
- **详细表格**：每个查询的评分明细
- **对比视图**：本次 vs 上次的指标差异
- **拒答分析**：拒答率、拒答原因分布

### 3. GitHub Actions 工作流
- 触发条件：`iteration7/corpus/*.txt` 文件变更
- 自动安装依赖、运行评估、生成报告
- 结果自动提交回仓库

### 4. 版本对比机制
- 维护 baseline.json 作为参考基准
- 每次运行自动与 baseline 和上次结果对比
- 高亮退化指标（红色）和改进指标（绿色）

## 验收标准

- [ ] 在 corpus 下添加新文档，push 后自动触发 workflow
- [ ] Workflow 在 5 分钟内完成，无错误
- [ ] `data/` 目录生成新的结果 JSON 文件
- [ ] `reports/index.html` 更新，通过 GitHub Pages 可访问
- [ ] HTML 报告显示完整指标和趋势图
- [ ] 修改代码（如改 embedding 模型），能检测到性能变化

## 产出物

1. 完整的 CI 式评估闭环
2. 可视化的历史趋势报告
3. 扩充到 100+ 查询的测试集
4. 迭代到第 7 轮后的最终系统分数报告

## 技术细节

### 成本控制
- 限制 workflow 并发（单实例运行）
- 缓存 embedding 结果（复用 chroma_db）
- 只在必要时触发（corpus 或核心代码变更）

### 安全性
- API 密钥通过 GitHub Secrets 管理
- 不在日志中暴露敏感信息

### 可维护性
- 模块化设计：评估、报告生成、可视化分离
- 配置文件驱动（rejection_config.json 等）
- 清晰的错误处理和日志

## 后续优化方向

1. **测试集质量管理**：定期清理低质量查询
2. **A/B 测试**：同时评估多个配置，选择最优
3. **告警机制**：指标退化时发送通知
4. **性能优化**：并行处理、增量评估
