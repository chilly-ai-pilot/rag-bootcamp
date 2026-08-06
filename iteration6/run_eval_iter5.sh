#!/bin/bash

# Iteration 5 完整评估脚本（使用异步批处理加速）
# 
# 使用方法:
#   bash run_full_eval.sh                    # 默认: hybrid, no rerank, LLM judge
#   bash run_full_eval.sh hybrid bge llm 10  # hybrid + bge rerank + LLM judge
#   bash run_full_eval.sh vector none llm 10 # vector only + LLM judge
#   bash run_full_eval.sh hybrid bge ragas 5 # hybrid + bge rerank + Ragas judge

RETRIEVAL_MODE=${1:-hybrid}
RERANK_MODE=${2:-none}
JUDGE_MODE=${3:-llm}
BATCH_SIZE=${4:-10}

echo "======================================"
echo "Iteration 5 完整评估（异步批处理）"
echo "======================================"
echo "召回模式: $RETRIEVAL_MODE"
echo "重排序: $RERANK_MODE"
echo "Judge 模式: $JUDGE_MODE"
echo "批次大小: $BATCH_SIZE"
echo ""
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python3 run_eval.py \
    --chunking-strategy fixed_200_40 \
    --retrieval-mode $RETRIEVAL_MODE \
    --rerank-mode $RERANK_MODE \
    --judge-mode $JUDGE_MODE \
    --batch-size $BATCH_SIZE

echo ""
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 根据配置生成文件名
if [ "$RERANK_MODE" != "none" ]; then
    OUTPUT_FILE="results_fixed_200_40_${RETRIEVAL_MODE}_rerank_${RERANK_MODE}.json"
else
    OUTPUT_FILE="results_fixed_200_40_${RETRIEVAL_MODE}.json"
fi

echo "✅ 评估完成！结果已保存到 $OUTPUT_FILE"
echo ""
echo "下一步：运行分析脚本"
echo "  python3 analyze_faithfulness.py $OUTPUT_FILE"
