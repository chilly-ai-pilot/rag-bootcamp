"""
下载 bge-reranker-base 模型

使用 HuggingFace 镜像加速下载
"""
import os

# 设置 HuggingFace 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("=" * 60)
print("下载 bge-reranker-base 模型")
print("=" * 60)
print(f"\n使用镜像: {os.environ['HF_ENDPOINT']}")
print("预计大小: ~1.2 GB")
print("下载位置: ~/.cache/huggingface/hub/\n")

try:
    from FlagEmbedding import FlagReranker
    
    print("正在下载模型，请稍候...")
    print("(首次下载可能需要几分钟，后续会使用缓存)\n")
    
    reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
    
    print("\n" + "=" * 60)
    print("✅ 模型下载成功！")
    print("=" * 60)
    
    # 测试模型
    print("\n测试模型...")
    test_pairs = [
        ["你好", "Hello"],
        ["你好", "再见"]
    ]
    scores = reranker.compute_score(test_pairs)
    print(f"测试分数: {scores}")
    print("\n✅ 模型工作正常！")
    
    print("\n现在可以运行:")
    print("  python3 test_reranker.py")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ 下载失败")
    print("=" * 60)
    print(f"\n错误: {e}")
    print("\n解决方案:")
    print("1. 检查网络连接")
    print("2. 尝试使用VPN")
    print("3. 手动下载:")
    print("   访问: https://hf-mirror.com/BAAI/bge-reranker-base")
    print("   下载所有文件到: ~/.cache/huggingface/hub/models--BAAI--bge-reranker-base/")
