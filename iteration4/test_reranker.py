"""
Reranker 功能测试脚本

用于验证：
1. FlagEmbedding 库是否正确安装
2. bge-reranker-base 模型能否成功加载
3. Rerank 功能是否正常工作
4. 分数范围和排序逻辑是否符合预期

使用方法:
    python3 test_reranker.py
"""

print("=" * 60)
print("Reranker 功能测试")
print("=" * 60)

# Test 1: 导入测试
print("\n[Test 1] 检查依赖导入...")
try:
    from FlagEmbedding import FlagReranker
    print("✅ FlagEmbedding 导入成功")
except ImportError as e:
    print(f"❌ FlagEmbedding 导入失败: {e}")
    print("请运行: pip install FlagEmbedding")
    exit(1)

# Test 2: 模型加载测试
print("\n[Test 2] 加载 bge-reranker-base 模型...")
try:
    reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
    print("✅ Reranker 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit(1)

# Test 3: 基本功能测试
print("\n[Test 3] 测试 Rerank 基本功能...")
query = "SmartLock-100 如何生成临时密码？"
docs = [
    "SmartLock-100 可通过 App 生成一次性或限时临时密码，适用于访客或家政服务。",  # 相关
    "SmartLock-200 支持人脸识别和 3D 结构光活体检测功能。",  # 不相关（型号不对）
    "智能门锁安装指南：请确认开门方向（左开/右开/内开/外开）。",  # 不相关（安装信息）
    "SmartLock-100 指纹容量 100 枚，识别速度小于 0.3 秒。",  # 相关但不是答案
]

try:
    # 计算 rerank 分数
    pairs = [[query, doc] for doc in docs]
    scores = reranker.compute_score(pairs, normalize=True)
    
    print(f"\nQuery: {query}\n")
    print("Rerank 结果:")
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"  [{i+1}] Score: {score:.4f}")
        print(f"      Doc: {doc[:60]}...")
    
    # 验证分数范围
    assert all(0 <= s <= 1 for s in scores), "分数应该在 [0, 1] 范围内"
    print("\n✅ 分数范围正常 (0-1)")
    
    # 验证排序逻辑
    best_idx = scores.index(max(scores))
    assert best_idx == 0, f"预期第 1 个文档得分最高，但实际是第 {best_idx+1} 个"
    print("✅ 排序逻辑正确（相关文档得分最高）")
    
except Exception as e:
    print(f"❌ Rerank 功能测试失败: {e}")
    exit(1)

# Test 4: 与 Hybrid 检索集成测试
print("\n[Test 4] 测试与检索系统集成...")
try:
    from chunking import build_corpus_chunks
    from retrieval import retrieve_hybrid, retrieve_rerank
    
    # 构建小规模测试语料
    chunks = build_corpus_chunks("corpus", strategy="small_100_50")
    print(f"✅ 加载了 {len(chunks)} 个文档块")
    
    # 测试 hybrid 检索
    test_query = "SmartLock-100 如何生成临时密码？"
    hybrid_results = retrieve_hybrid(test_query, chunks, k=5, strategy="small_100_50")
    print(f"✅ Hybrid 检索返回 {len(hybrid_results)} 个结果")
    
    # 测试 rerank 检索
    rerank_results = retrieve_rerank(test_query, chunks, k=5, strategy="small_100_50", k_candidates=20)
    print(f"✅ Rerank 检索返回 {len(rerank_results)} 个结果")
    
    # 验证 rerank_score 字段
    assert all('rerank_score' in r for r in rerank_results), "结果应包含 rerank_score 字段"
    print("✅ Rerank 结果包含分数字段")
    
    # 打印对比
    print("\n前3个结果对比:")
    print("\nHybrid 检索:")
    for i, r in enumerate(hybrid_results[:3], 1):
        print(f"  [{i}] {r['text'][:60]}...")
    
    print("\nRerank 检索:")
    for i, r in enumerate(rerank_results[:3], 1):
        print(f"  [{i}] Score: {r['rerank_score']:.4f}")
        print(f"      {r['text'][:60]}...")
    
except Exception as e:
    print(f"❌ 集成测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 全部测试通过
print("\n" + "=" * 60)
print("🎉 所有测试通过！Reranker 功能正常！")
print("=" * 60)
print("\n下一步:")
print("  运行完整评估: python3 run_eval.py --chunking-strategy small_100_50 --retrieval-mode rerank")
