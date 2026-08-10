"""
测试自愈机制

模拟触发条件并生成审核文件
"""

import json
from self_healing import (
    should_trigger_self_healing,
    create_review_file,
    deduplicate_review_files,
    list_pending_reviews,
    process_self_healing
)


def test_trigger_conditions():
    """测试触发条件"""
    print("="*60)
    print("测试 1: 触发条件检测")
    print("="*60)
    
    self_healing_config = {
        "enabled": True,
        "triggers": {
            "hit_not_1": True,
            "answer_rank_threshold": 4,
            "layer1_rejection": True
        }
    }
    
    # 测试 1: hit != 1
    result1 = {"hit": 0, "answer_rank": None}
    should, reason = should_trigger_self_healing(result1, self_healing_config)
    print(f"\n测试 hit=0: {should} (reason: {reason})")
    assert should == True
    assert reason == "retrieval_miss"
    
    # 测试 2: answer_rank > 4
    result2 = {"hit": 1, "answer_rank": 5}
    should, reason = should_trigger_self_healing(result2, self_healing_config)
    print(f"测试 rank=5: {should} (reason: {reason})")
    assert should == True
    assert reason == "low_rank_5"
    
    # 测试 3: layer1_rejection
    result3 = {
        "hit": 1,
        "answer_rank": 2,
        "rejected": True,
        "rejection_reason": "[Layer 1] Top-1 rerank score too low"
    }
    should, reason = should_trigger_self_healing(result3, self_healing_config)
    print(f"测试 layer1拒答: {should} (reason: {reason})")
    assert should == True
    assert reason == "layer1_rejection"
    
    # 测试 4: 不触发
    result4 = {"hit": 1, "answer_rank": 2, "rejected": False}
    should, reason = should_trigger_self_healing(result4, self_healing_config)
    print(f"测试 正常情况: {should} (reason: {reason})")
    assert should == False
    
    print("\n✅ 所有触发条件测试通过")


def test_create_review():
    """测试创建审核文件"""
    print("\n" + "="*60)
    print("测试 2: 创建审核文件")
    print("="*60)
    
    filepath = create_review_file(
        query="测试查询：SmartLock-100 的功能",
        ground_truth="SmartLock-100 支持指纹、密码、卡片等多种开锁方式。",
        doc_id="doc1",
        char_start=100,
        char_end=200,
        trigger_reason="retrieval_miss",
        rejection_reason=None,
        review_dir="review"
    )
    
    print(f"\n✅ 创建审核文件: {filepath}")
    
    # 验证文件内容
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert data['query'] == "测试查询：SmartLock-100 的功能"
    assert data['status'] == "pending"
    assert 'query_hash' in data
    
    print(f"   Query hash: {data['query_hash']}")
    print(f"   Status: {data['status']}")


def test_deduplicate():
    """测试去重"""
    print("\n" + "="*60)
    print("测试 3: 去重审核文件")
    print("="*60)
    
    # 创建两个相同 query 的文件
    query = "重复查询测试"
    
    create_review_file(
        query=query,
        ground_truth="答案1",
        doc_id="doc1",
        char_start=0,
        char_end=10,
        trigger_reason="test",
        rejection_reason=None,
        review_dir="review"
    )
    
    import time
    time.sleep(0.1)  # 确保时间戳不同
    
    create_review_file(
        query=query,
        ground_truth="答案2（较新）",
        doc_id="doc1",
        char_start=0,
        char_end=10,
        trigger_reason="test",
        rejection_reason=None,
        review_dir="review"
    )
    
    print(f"\n创建了 2 个重复文件")
    
    # 去重
    deleted_count = deduplicate_review_files("review")
    print(f"✅ 删除了 {deleted_count} 个重复文件")
    
    # 验证只剩一个
    pending = list_pending_reviews("review")
    matching = [r for r in pending if r['query'] == query]
    assert len(matching) == 1
    print(f"   保留了最新的文件")


def test_list_pending():
    """测试列出待审核"""
    print("\n" + "="*60)
    print("测试 4: 列出待审核文件")
    print("="*60)
    
    pending = list_pending_reviews("review")
    
    print(f"\n找到 {len(pending)} 个待审核文件:")
    for i, review in enumerate(pending, 1):
        print(f"\n{i}. Query: {review['query']}")
        print(f"   Reason: {review['trigger_reason']}")
        print(f"   Status: {review['status']}")
        print(f"   File: {review['filename']}")


def test_full_process():
    """测试完整流程"""
    print("\n" + "="*60)
    print("测试 5: 完整自愈流程")
    print("="*60)
    
    # 模拟评估结果
    results = [
        {
            "id": 1,
            "query": "SmartLock-100 如何生成临时密码？",
            "category": "chunking_sensitive",
            "hit": 0,  # 触发 hit_not_1
            "answer_rank": None
        },
        {
            "id": 2,
            "query": "SmartCam-200 的夜视距离？",
            "category": "exact_match",
            "hit": 1,
            "answer_rank": 5,  # 触发 answer_rank > 4
        },
        {
            "id": 3,
            "query": "网关的指示灯含义？",
            "category": "semantic_paraphrase",
            "hit": 1,
            "answer_rank": 2,
            "rejected": True,
            "rejection_reason": "[Layer 1] Top-1 rerank score too low"  # 触发 layer1_rejection
        },
        {
            "id": 4,
            "query": "SmartBulb-300 的功率？",
            "category": "exact_match",
            "hit": 1,
            "answer_rank": 1,  # 正常，不触发
            "rejected": False
        }
    ]
    
    # 模拟查询数据
    queries = [
        {
            "id": 1,
            "query": "SmartLock-100 如何生成临时密码？",
            "doc_id": "doc1",
            "ground_truth_text": "可通过 App 生成一次性或限时临时密码。",
            "char_start": 700,
            "char_end": 750
        },
        {
            "id": 2,
            "query": "SmartCam-200 的夜视距离？",
            "doc_id": "doc2",
            "ground_truth_text": "红外夜视距离可达 10 米。",
            "char_start": 90,
            "char_end": 110
        },
        {
            "id": 3,
            "query": "网关的指示灯含义？",
            "doc_id": "doc4",
            "ground_truth_text": "红灯常亮：未连接网络；蓝灯常亮：工作正常。",
            "char_start": 115,
            "char_end": 145
        },
        {
            "id": 4,
            "query": "SmartBulb-300 的功率？",
            "doc_id": "doc3",
            "ground_truth_text": "功率 9W。",
            "char_start": 50,
            "char_end": 60
        }
    ]
    
    # 自愈配置
    self_healing_config = {
        "enabled": True,
        "triggers": {
            "hit_not_1": True,
            "answer_rank_threshold": 4,
            "layer1_rejection": True
        },
        "review_dir": "review",
        "auto_deduplicate": True
    }
    
    # 处理自愈
    stats = process_self_healing(results, queries, self_healing_config)
    
    print(f"\n✅ 自愈处理完成:")
    print(f"   触发次数: {stats['triggered_count']}")
    print(f"   创建文件: {len(stats['created_files'])}")
    print(f"   去重删除: {stats['deduplicated_count']}")
    
    # 应该触发 3 次（Q1, Q2, Q3）
    assert stats['triggered_count'] == 3


def main():
    """运行所有测试"""
    print("开始测试自愈机制...")
    
    test_trigger_conditions()
    test_create_review()
    test_deduplicate()
    test_list_pending()
    test_full_process()
    
    print("\n" + "="*60)
    print("✅ 所有测试通过！")
    print("="*60)


if __name__ == "__main__":
    main()
