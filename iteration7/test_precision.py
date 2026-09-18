"""
测试 Precision@K 计算功能
"""
from scoring import calculate_precision_at_k

# 测试数据
test_results = [
    {
        'category': 'test',
        'doc_id': 'doc1',
        'char_start': 100,
        'char_end': 200,
        'retrieved': [
            {'doc_id': 'doc1', 'start': 90, 'end': 150, 'text': 'relevant chunk 1'},  # 重叠，相关
            {'doc_id': 'doc1', 'start': 150, 'end': 250, 'text': 'relevant chunk 2'},  # 重叠，相关
            {'doc_id': 'doc2', 'start': 0, 'end': 100, 'text': 'irrelevant chunk 1'},  # 不相关
            {'doc_id': 'doc2', 'start': 100, 'end': 200, 'text': 'irrelevant chunk 2'},  # 不相关
            {'doc_id': 'doc1', 'start': 300, 'end': 400, 'text': 'irrelevant chunk 3'},  # 不相关（不重叠）
        ]
    },
    {
        'category': 'test',
        'doc_id': 'doc3',
        'char_start': 0,
        'char_end': 50,
        'retrieved': [
            {'doc_id': 'doc3', 'start': 0, 'end': 100, 'text': 'relevant chunk'},  # 重叠，相关
            {'doc_id': 'doc4', 'start': 0, 'end': 100, 'text': 'irrelevant chunk 1'},
            {'doc_id': 'doc4', 'start': 100, 'end': 200, 'text': 'irrelevant chunk 2'},
            {'doc_id': 'doc4', 'start': 200, 'end': 300, 'text': 'irrelevant chunk 3'},
            {'doc_id': 'doc4', 'start': 300, 'end': 400, 'text': 'irrelevant chunk 4'},
        ]
    }
]

# 测试 Precision@5
print("=" * 60)
print("Testing Precision@K Calculation")
print("=" * 60)

k = 5
precision_scores = calculate_precision_at_k(test_results, k)

print(f"\n✅ Precision@{k} Results:")
for category, score in precision_scores.items():
    print(f"  {category}: {score:.4f}")

# 验证结果
# Query 1: 前5个中有2个相关 = 2/5 = 0.4
# Query 2: 前5个中有1个相关 = 1/5 = 0.2
# Overall: (0.4 + 0.2) / 2 = 0.3

expected_overall = 0.3
actual_overall = precision_scores['overall']

print(f"\n📊 Validation:")
print(f"  Expected overall: {expected_overall:.4f}")
print(f"  Actual overall:   {actual_overall:.4f}")

if abs(expected_overall - actual_overall) < 0.0001:
    print("  ✅ Test PASSED!")
else:
    print("  ❌ Test FAILED!")
    exit(1)

# 测试 Precision@3
print(f"\n{'=' * 60}")
k = 3
precision_scores = calculate_precision_at_k(test_results, k)

print(f"\n✅ Precision@{k} Results:")
for category, score in precision_scores.items():
    print(f"  {category}: {score:.4f}")

# Query 1: 前3个中有2个相关 = 2/3 = 0.6667
# Query 2: 前3个中有1个相关 = 1/3 = 0.3333
# Overall: (0.6667 + 0.3333) / 2 = 0.5

expected_overall = 0.5
actual_overall = precision_scores['overall']

print(f"\n📊 Validation:")
print(f"  Expected overall: {expected_overall:.4f}")
print(f"  Actual overall:   {actual_overall:.4f}")

if abs(expected_overall - actual_overall) < 0.0001:
    print("  ✅ Test PASSED!")
else:
    print("  ❌ Test FAILED!")
    exit(1)

print(f"\n{'=' * 60}")
print("✅ All tests passed!")
print("=" * 60)
