#!/bin/bash

# Iteration 4 运行脚本
# 按顺序执行：安装依赖 → 测试 → baseline → rerank 评估

echo "========================================"
echo "Iteration 4: Reranker 评估流程"
echo "========================================"

# Step 1: 检查并安装依赖
echo ""
echo "[Step 1] 检查依赖..."
pip3 list | grep -q FlagEmbedding
if [ $? -ne 0 ]; then
    echo "⚠️  FlagEmbedding 未安装，正在安装..."
    pip3 install FlagEmbedding
else
    echo "✅ FlagEmbedding 已安装"
fi

# Step 2: 运行 Reranker 功能测试
echo ""
echo "[Step 2] 运行 Reranker 功能测试..."
python3 test_reranker.py
if [ $? -ne 0 ]; then
    echo "❌ 测试失败，请检查错误信息"
    exit 1
fi

# Step 3: 运行 Hybrid baseline（如果结果文件不存在）
echo ""
echo "[Step 3] 检查 Hybrid baseline..."
if [ ! -f "results_small_100_50_hybrid.json" ]; then
    echo "⚠️  Baseline 结果不存在，正在运行..."
    python3 run_eval.py --chunking-strategy small_100_50 --retrieval-mode hybrid
else
    echo "✅ Baseline 结果已存在: results_small_100_50_hybrid.json"
fi

# Step 4: 运行 Rerank 评估
echo ""
echo "[Step 4] 运行 Rerank 评估..."
python3 run_eval.py --chunking-strategy small_100_50 --retrieval-mode rerank

# 完成
echo ""
echo "========================================"
echo "✅ Iteration 4 评估完成！"
echo "========================================"
echo ""
echo "生成的文件:"
echo "  - results_small_100_50_hybrid.json (baseline)"
echo "  - results_small_100_50_rerank.json (rerank)"
echo ""
echo "下一步："
echo "  python3 compare_results.py  # 对比结果"
