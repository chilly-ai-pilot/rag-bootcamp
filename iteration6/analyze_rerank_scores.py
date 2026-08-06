#!/usr/bin/env python3
"""
分析 Rerank 分数分布

功能：
1. 分析所有 rerank 分数的统计特征
2. 对比命中 vs 未命中查询的分数分布
3. 验证阈值建议的合理性
4. 为 Iteration 6 拒答机制提供数据支持

使用方法:
    python3 analyze_rerank_scores.py
"""

import json
import statistics
from typing import List, Dict, Any


def load_results(filepath: str) -> Dict[str, Any]:
    """加载结果文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_score_distribution(results: List[Dict]):
    """分析分数分布"""
    print("=" * 80)
    print("📊 Rerank 分数整体分布")
    print("=" * 80)
    print()
    
    # 收集所有分数
    all_scores = []
    for result in results:
        if 'rerank_scores' in result:
            all_scores.extend(result['rerank_scores'])
    
    if not all_scores:
        print("❌ 未找到 rerank_scores 数据")
        return
    
    # 统计特征
    all_scores_sorted = sorted(all_scores)
    n = len(all_scores)
    
    print(f"总分数数量:     {n}")
    print(f"范围:           [{min(all_scores):.4f}, {max(all_scores):.4f}]")
    print(f"均值:           {statistics.mean(all_scores):.4f}")
    print(f"中位数:         {statistics.median(all_scores):.4f}")
    print(f"标准差:         {statistics.stdev(all_scores):.4f}")
    print()
    
    # 分位数
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("分位数分布:")
    for p in percentiles:
        idx = int(n * p / 100)
        value = all_scores_sorted[min(idx, n-1)]
        print(f"  P{p:2d}:  {value:.4f}")
    print()
    
    # 分数区间分布
    print("分数区间分布:")
    bins = [
        (0.0, 0.3, "极低 (0.0-0.3)"),
        (0.3, 0.5, "较低 (0.3-0.5)"),
        (0.5, 0.7, "中等 (0.5-0.7)"),
        (0.7, 0.9, "较高 (0.7-0.9)"),
        (0.9, 1.0, "极高 (0.9-1.0)"),
    ]
    
    for low, high, label in bins:
        count = sum(1 for s in all_scores if low <= s < high)
        pct = count / n * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:<20} {count:>4} ({pct:>5.1f}%) {bar}")
    
    # 特别关注高分区间
    high_scores = [s for s in all_scores if s >= 0.95]
    print(f"\n高分数（≥0.95）占比: {len(high_scores)} / {n} ({len(high_scores)/n*100:.1f}%)")
    print()


def analyze_by_hit_status(results: List[Dict]):
    """按命中/未命中分析分数"""
    print("=" * 80)
    print("🎯 命中 vs 未命中查询的分数对比")
    print("=" * 80)
    print()
    
    # 分组
    hit_queries = []
    miss_queries = []
    
    for result in results:
        if 'rerank_scores' not in result:
            continue
        
        top1_score = result['rerank_scores'][0] if result['rerank_scores'] else 0
        
        if result['hit'] == 1:
            hit_queries.append({
                'id': result['id'],
                'query': result['query'],
                'category': result['category'],
                'rank': result.get('answer_rank', 999),
                'top1_score': top1_score,
                'all_scores': result['rerank_scores']
            })
        else:
            miss_queries.append({
                'id': result['id'],
                'query': result['query'],
                'category': result['category'],
                'top1_score': top1_score,
                'all_scores': result['rerank_scores']
            })
    
    print(f"命中查询数:   {len(hit_queries)}")
    print(f"未命中查询数: {len(miss_queries)}")
    print()
    
    # 命中查询的 top-1 分数分析
    if hit_queries:
        hit_top1_scores = [q['top1_score'] for q in hit_queries]
        print("【命中查询 - Top-1 分数】")
        print(f"  均值:   {statistics.mean(hit_top1_scores):.4f}")
        print(f"  中位数: {statistics.median(hit_top1_scores):.4f}")
        print(f"  范围:   [{min(hit_top1_scores):.4f}, {max(hit_top1_scores):.4f}]")
        
        # 分位数
        hit_sorted = sorted(hit_top1_scores)
        n_hit = len(hit_sorted)
        p10_hit = hit_sorted[int(n_hit * 0.10)]
        p25_hit = hit_sorted[int(n_hit * 0.25)]
        p50_hit = hit_sorted[int(n_hit * 0.50)]
        
        print(f"  P10:    {p10_hit:.4f}")
        print(f"  P25:    {p25_hit:.4f}")
        print(f"  P50:    {p50_hit:.4f}")
        print()
    
    # 未命中查询的 top-1 分数分析
    if miss_queries:
        miss_top1_scores = [q['top1_score'] for q in miss_queries]
        print("【未命中查询 - Top-1 分数】")
        print(f"  均值:   {statistics.mean(miss_top1_scores):.4f}")
        print(f"  中位数: {statistics.median(miss_top1_scores):.4f}")
        print(f"  范围:   [{min(miss_top1_scores):.4f}, {max(miss_top1_scores):.4f}]")
        
        # 分位数
        miss_sorted = sorted(miss_top1_scores)
        n_miss = len(miss_sorted)
        p50_miss = miss_sorted[int(n_miss * 0.50)]
        p75_miss = miss_sorted[int(n_miss * 0.75)]
        p90_miss = miss_sorted[int(n_miss * 0.90)]
        
        print(f"  P50:    {p50_miss:.4f}")
        print(f"  P75:    {p75_miss:.4f}")
        print(f"  P90:    {p90_miss:.4f}")
        print()
    
    # 分布对比可视化
    print("【Top-1 分数分布对比】")
    print()
    
    bins = [
        (0.0, 0.5, "0.0-0.5"),
        (0.5, 0.7, "0.5-0.7"),
        (0.7, 0.8, "0.7-0.8"),
        (0.8, 0.9, "0.8-0.9"),
        (0.9, 0.95, "0.9-0.95"),
        (0.95, 1.0, "0.95-1.0"),
    ]
    
    print(f"{'区间':<12} {'命中':<20} {'未命中':<20}")
    print("-" * 60)
    
    for low, high, label in bins:
        if hit_queries:
            hit_count = sum(1 for q in hit_queries if low <= q['top1_score'] < high)
            hit_pct = hit_count / len(hit_queries) * 100
            hit_bar = "█" * int(hit_pct / 5)
            hit_display = f"{hit_count:>2} ({hit_pct:>5.1f}%) {hit_bar}"
        else:
            hit_display = "-"
        
        if miss_queries:
            miss_count = sum(1 for q in miss_queries if low <= q['top1_score'] < high)
            miss_pct = miss_count / len(miss_queries) * 100
            miss_bar = "█" * int(miss_pct / 5)
            miss_display = f"{miss_count:>2} ({miss_pct:>5.1f}%) {miss_bar}"
        else:
            miss_display = "-"
        
        print(f"{label:<12} {hit_display:<20} {miss_display:<20}")
    
    print()


def analyze_threshold_effectiveness(results: List[Dict], thresholds: List[float]):
    """分析不同阈值的效果"""
    print("=" * 80)
    print("🔍 阈值效果分析（为 Iteration 6 做准备）")
    print("=" * 80)
    print()
    
    print("如果使用阈值进行拒答，模拟效果如下：")
    print()
    print(f"{'阈值':<10} {'拒答数':<12} {'错误拒答':<12} {'正确拒答':<12} {'准确率':<12} {'评价'}")
    print("-" * 80)
    
    for threshold in thresholds:
        total_reject = 0
        false_reject = 0  # 本该回答但拒答了（命中查询被拒）
        true_reject = 0   # 正确拒答（未命中查询被拒）
        
        for result in results:
            if 'rerank_scores' not in result or not result['rerank_scores']:
                continue
            
            top1_score = result['rerank_scores'][0]
            
            if top1_score < threshold:
                total_reject += 1
                if result['hit'] == 1:
                    false_reject += 1
                else:
                    true_reject += 1
        
        if total_reject > 0:
            accuracy = true_reject / total_reject
        else:
            accuracy = 1.0
        
        # 评价
        if accuracy >= 0.9 and false_reject <= 2:
            verdict = "✅ 推荐"
        elif accuracy >= 0.8:
            verdict = "😐 可用"
        elif accuracy >= 0.5:
            verdict = "⚠️ 谨慎"
        else:
            verdict = "❌ 不建议"
        
        print(f"{threshold:<10.4f} {total_reject:<12} {false_reject:<12} {true_reject:<12} "
              f"{accuracy:<12.1%} {verdict}")
    
    print()
    print("说明:")
    print("  - 拒答数: 低于阈值被拒绝回答的查询数")
    print("  - 错误拒答: 本该回答（命中）但被拒答的查询数")
    print("  - 正确拒答: 正确拒绝（未命中）的查询数")
    print("  - 准确率: 正确拒答 / 总拒答数")
    print()


def show_edge_cases(results: List[Dict]):
    """展示边界案例"""
    print("=" * 80)
    print("🔬 边界案例分析")
    print("=" * 80)
    print()
    
    # 找出命中但分数低的案例
    hit_low_score = []
    for result in results:
        if result['hit'] == 1 and 'rerank_scores' in result and result['rerank_scores']:
            top1_score = result['rerank_scores'][0]
            if top1_score < 0.8:
                hit_low_score.append({
                    'id': result['id'],
                    'query': result['query'],
                    'category': result['category'],
                    'rank': result.get('answer_rank', 999),
                    'score': top1_score
                })
    
    if hit_low_score:
        print(f"【命中但分数较低的案例】({len(hit_low_score)} 个)")
        print("（这些案例如果设置阈值过高会被误拒）")
        print()
        for item in sorted(hit_low_score, key=lambda x: x['score']):
            print(f"Query {item['id']}: {item['query']}")
            print(f"  类型: {item['category']}, 排名: {item['rank']}, Top-1分数: {item['score']:.4f}")
            print()
    else:
        print("✅ 没有命中但分数较低的案例（所有命中查询分数都 ≥0.8）")
        print()
    
    # 找出未命中但分数高的案例
    miss_high_score = []
    for result in results:
        if result['hit'] == 0 and 'rerank_scores' in result and result['rerank_scores']:
            top1_score = result['rerank_scores'][0]
            if top1_score >= 0.9:
                miss_high_score.append({
                    'id': result['id'],
                    'query': result['query'],
                    'category': result['category'],
                    'score': top1_score
                })
    
    if miss_high_score:
        print(f"【未命中但分数较高的案例】({len(miss_high_score)} 个)")
        print("（这些案例 reranker 认为相关，但实际答案不在 top-5）")
        print()
        for item in sorted(miss_high_score, key=lambda x: -x['score']):
            print(f"Query {item['id']}: {item['query']}")
            print(f"  类型: {item['category']}, Top-1分数: {item['score']:.4f}")
            print()
    else:
        print("✅ 没有未命中但分数较高的案例（未命中查询分数都 <0.9）")
        print()


def main():
    """主函数"""
    print()
    print("=" * 80)
    print("📈 Rerank 分数分布分析")
    print("=" * 80)
    print()
    
    # 加载结果
    print("📂 加载结果文件...")
    try:
        data = load_results('results_fixed_100_50_rerank.json')
        results = data['results']
        print(f"  ✅ 加载成功: {len(results)} 个查询")
    except FileNotFoundError:
        print("  ❌ 未找到 results_fixed_100_50_rerank.json")
        print("  请先运行: python3 run_eval.py --chunking-strategy fixed_100_50 --retrieval-mode rerank")
        return
    
    print()
    
    # 分析
    analyze_score_distribution(results)
    analyze_by_hit_status(results)
    
    # 测试不同阈值
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    analyze_threshold_effectiveness(results, thresholds)
    
    # 边界案例
    show_edge_cases(results)
    
    # 总结建议
    print("=" * 80)
    print("📝 阈值建议总结")
    print("=" * 80)
    print()
    
    print("基于以上分析，为 Iteration 6 拒答机制提供以下建议：")
    print()
    print("1️⃣ 保守策略（降低误答，但可能漏答）:")
    print("   阈值: 0.95")
    print("   特点: 只拒绝明确不相关的查询，几乎不会误拒")
    print("   适用: 对准确性要求极高的场景")
    print()
    print("2️⃣ 推荐策略（平衡准确率和召回率）:")
    print("   阈值: 0.85-0.90")
    print("   特点: 在保持高准确率的同时，有效过滤低质量结果")
    print("   适用: 大多数生产场景")
    print()
    print("3️⃣ 激进策略（降低漏答，但可能误答）:")
    print("   阈值: 0.70-0.80")
    print("   特点: 更多回答，但可能包含不太相关的内容")
    print("   适用: 召回率优先的场景")
    print()
    print("💡 实施建议:")
    print("  - 从推荐策略开始（阈值 0.85-0.90）")
    print("  - 在真实数据上 A/B 测试，根据用户反馈调整")
    print("  - 考虑根据查询类型使用不同阈值")
    print("  - 记录被拒查询，定期分析改进")
    print()


if __name__ == "__main__":
    main()
