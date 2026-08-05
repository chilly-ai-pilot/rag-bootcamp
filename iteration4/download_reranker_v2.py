#!/usr/bin/env python3
"""
下载 bge-reranker-base 模型 (v2 - 使用 HuggingFace Hub)
"""

import os
import sys

# 设置环境变量（必须在导入库之前）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def download_model():
    """使用 huggingface_hub 下载模型"""
    
    print("=" * 70)
    print("  下载 bge-reranker-base 模型 (使用 HuggingFace Hub 方式)")
    print("=" * 70)
    print()
    print(f"镜像站点: {os.environ.get('HF_ENDPOINT', '默认')}")
    print("模型大小: ~1.2 GB")
    print("缓存位置: ~/.cache/huggingface/hub/")
    print()
    
    try:
        # 先尝试用 huggingface_hub
        print("[1/3] 导入 huggingface_hub...")
        from huggingface_hub import snapshot_download
        
        model_name = "BAAI/bge-reranker-base"
        
        print(f"[2/3] 下载模型: {model_name}")
        print("      (这可能需要几分钟，请耐心等待...)")
        print()
        
        # 下载整个模型仓库
        cache_dir = snapshot_download(
            repo_id=model_name,
            resume_download=True,  # 支持断点续传
            local_files_only=False
        )
        
        print()
        print("[3/3] 验证模型...")
        
        # 验证下载
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker(model_name, use_fp16=True)
        
        # 测试
        scores = reranker.compute_score([['测试', '测试文档']])
        
        print()
        print("=" * 70)
        print("✅ 成功！")
        print("=" * 70)
        print()
        print(f"模型缓存位置: {cache_dir}")
        print(f"测试分数: {scores}")
        print()
        print("现在可以运行:")
        print("  python3 test_reranker.py")
        print("  python3 run_eval.py --chunking-strategy small_100_50 --retrieval-mode rerank")
        print()
        
        return True
        
    except ImportError as e:
        print()
        print(f"❌ 缺少依赖: {e}")
        print()
        print("请安装: pip install huggingface_hub")
        return False
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ 下载失败")
        print("=" * 70)
        print()
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)[:500]}")
        print()
        print("可能的解决方案:")
        print()
        print("1. 检查网络连接")
        print("   - 确保可以访问 hf-mirror.com")
        print("   - 尝试在浏览器打开: https://hf-mirror.com/BAAI/bge-reranker-base")
        print()
        print("2. 使用代理或 VPN")
        print()
        print("3. 手动下载（如果自动下载失败）:")
        print("   a. 访问: https://hf-mirror.com/BAAI/bge-reranker-base/tree/main")
        print("   b. 下载所有文件（特别是 pytorch_model.bin, ~1.1GB）")
        print("   c. 放到: ~/.cache/huggingface/hub/models--BAAI--bge-reranker-base/")
        print("      snapshots/<hash>/")
        print()
        print("4. 如果已经下载了部分文件，删除重试:")
        print("   rm -rf ~/.cache/huggingface/hub/models--BAAI--bge-reranker-base")
        print()
        
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
