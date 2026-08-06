#!/bin/bash
# 快速运行 RAG 评估（包含 Faithfulness + Relevance 双重评估）

set -e

echo "🚀 开始运行 RAG 评估"
echo "================================"
echo ""

# 检查环境变量
if [ -z "$ALI_API_KEY" ] || [ -z "$ALI_BASE_URL" ]; then
    echo "❌ 错误: 请设置环境变量 ALI_API_KEY 和 ALI_BASE_URL"
    echo ""
    echo "示例:"
    echo "  export ALI_API_KEY='your-api-key'"
    echo "  export ALI_BASE_URL='your-base-url'"
    exit 1
fi

echo "✅ 环境变量已设置"
echo ""

# 运行评估
echo "📊 配置:"
echo "  - Chunking: fixed_100_50"
echo "  - Retrieval: vector"
echo "  - Judge: LLM (组合评估)"
echo "  - Batch size: 5"
echo ""

echo "⏳ 开始评估（32 个查询，预计 3-5 分钟）..."
echo ""

python3 run_eval.py \
  --chunking-strategy fixed_100_50 \
  --retrieval-mode vector \
  --judge-mode llm \
  --batch-size 10

echo ""
echo "================================"
echo "✅ 评估完成！"
echo ""
echo "📁 结果文件: results_fixed_100_50_vector.json"
echo ""
echo "💡 查看结果:"
echo "  python3 -c \"import json; print(json.dumps(json.load(open('results_fixed_100_50_vector.json')), indent=2)[:500])\""
echo ""
echo "📊 关键指标:"
python3 << 'EOF'
import json
import os

result_file = 'results_fixed_100_50_vector.json'
if os.path.exists(result_file):
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Recall@20: {data['scores']['overall']:.2%}")
    print(f"  MRR: {data['mrr_scores']['overall']:.4f}")
    
    if 'faithfulness_analysis' in data:
        fa = data['faithfulness_analysis']
        print(f"  Faithfulness: {fa['mean']:.3f} (median: {fa['median']:.3f})")
    
    if 'relevance_analysis' in data:
        ra = data['relevance_analysis']
        print(f"  Relevance: {ra['mean']:.3f} (median: {ra['median']:.3f})")
else:
    print("  (结果文件未找到)")
EOF

echo ""
echo "🎉 完成！"
