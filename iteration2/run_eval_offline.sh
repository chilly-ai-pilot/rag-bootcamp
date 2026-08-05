#!/bin/bash
# 强制使用本地缓存的模型，避免重复下载
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 run_eval.py "$@"
