"""
Iteration 5: Faithfulness 深度分析脚本

分析内容：
1. Faithfulness 与 Recall/MRR/Rerank 的相关性
2. 低 Faithfulness 案例诊断
3. 不同查询类别的 Faithfulness 表现
4. 为 Iteration 6 的拒答阈值提供建议
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List
import numpy as np


def load_results(result_file: str) -> Dict:
    """加载评估结果"""
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_correlation(results: List[Dict]) -> Dict:
    """
    分析 Faithfulness 与其他指标的相关性
    
    返回：
        {
            'faithfulness_vs_hit': correlation,
            'faithfulness_vs_rerank_top1': correlation,
            'faithfulness_vs_answer_rank': correlation
        }
    """
    print("\n" + "="*80)
    print("📊 相关性分析：Faithfulness 与其他指标")
    print("="*80)
    
    # 提取数据
    faithfulness_scores = []
    hit_flags = []
    rerank_top1_scores = []
    answer_ranks = []
    
    for r in results:
        f_score = r.get('faithfulness_score')
        if f_score is None:
            continue
        
        faithfulness_scores.append(f_score)
        hit_flags.append(1 if r.get('hit') else 0)
        
        rerank_scores = r.get('rerank_scores', [])
        if rerank_scores:
            rerank_top1_scores.append(rerank_scores[0])
        
        answer_rank = r.get('answer_rank')
        if answer_rank:
            answer_ranks.append(answer_rank)
    
    correlations = {}
    
    # Faithfulness vs Hit (命中/未命中)
    if len(faithfulness_scores) == len(hit_flags):
        corr = np.corrcoef(faithfulness_scores, hit_flags)[0, 1]
        correlations['faithfulness_vs_hit'] = corr
        print(f"\n1️⃣  Faithfulness vs Hit (检索是否命中)")
        print(f"   相关系数: {corr:.3f}")
        if corr > 0.5:
            print(f"   ✅ 强正相关：检索命中时，Faithfulness 更高")
        elif corr > 0.2:
            print(f"   ⚠️  弱正相关：检索命中对 Faithfulness 有一定影响")
        else:
            print(f"   ❌ 几乎无关：检索命中不保证答案忠实")
    
    # Faithfulness vs Rerank Top-1 Score
    if faithfulness_scores and rerank_top1_scores and len(faithfulness_scores) == len(rerank_top1_scores):
        corr = np.corrcoef(faithfulness_scores, rerank_top1_scores)[0, 1]
        correlations['faithfulness_vs_rerank_top1'] = corr
        print(f"\n2️⃣  Faithfulness vs Rerank Top-1 分数")
        print(f"   相关系数: {corr:.3f}")
        if corr > 0.5:
            print(f"   ✅ 强正相关：Rerank 分数高时，Faithfulness 更高")
        elif corr > 0.2:
            print(f"   ⚠️  弱正相关：Rerank 分数对 Faithfulness 有一定预测性")
        else:
            print(f"   ❌ 几乎无关：Rerank 分数高不代表答案忠实")
    
    # Faithfulness vs Answer Rank (答案排名)
    if faithfulness_scores and answer_ranks and len(faithfulness_scores) == len(answer_ranks):
        # 排名越小（越靠前）越好，所以期望负相关
        corr = np.corrcoef(faithfulness_scores, answer_ranks)[0, 1]
        correlations['faithfulness_vs_answer_rank'] = corr
        print(f"\n3️⃣  Faithfulness vs Answer Rank (答案排名)")
        print(f"   相关系数: {corr:.3f}")
        if corr < -0.5:
            print(f"   ✅ 强负相关：答案排名越靠前，Faithfulness 越高")
        elif corr < -0.2:
            print(f"   ⚠️  弱负相关：答案排名对 Faithfulness 有一定影响")
        else:
            print(f"   ❌ 几乎无关：答案排名与 Faithfulness 无明显关系")
    
    return correlations


def analyze_by_category(results: List[Dict]) -> Dict:
    """按查询类别分析 Faithfulness"""
    print("\n" + "="*80)
    print("📂 分类分析：不同查询类别的 Faithfulness")
    print("="*80)
    
    by_category = defaultdict(list)
    
    for r in results:
        category = r.get('category', 'unknown')
        f_score = r.get('faithfulness_score')
        if f_score is not None:
            by_category[category].append({
                'id': r['id'],
                'query': r['query'],
                'score': f_score,
                'hit': r.get('hit', False)
            })
    
    category_stats = {}
    
    for category, items in sorted(by_category.items()):
        scores = [item['score'] for item in items]
        hit_count = sum(1 for item in items if item['hit'])
        
        stats = {
            'count': len(scores),
            'mean': np.mean(scores),
            'median': np.median(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'std': np.std(scores),
            'hit_rate': hit_count / len(scores) if scores else 0
        }
        
        category_stats[category] = stats
        
        print(f"\n📁 {category}")
        print(f"   查询数量:  {stats['count']}")
        print(f"   平均分:    {stats['mean']:.3f}")
        print(f"   中位数:    {stats['median']:.3f}")
        print(f"   范围:      [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"   标准差:    {stats['std']:.3f}")
        print(f"   命中率:    {stats['hit_rate']:.1%}")
        
        # 找出该类别的低分案例
        low_score_items = [item for item in items if item['score'] < 0.5]
        if low_score_items:
            print(f"   ⚠️  低分案例 ({len(low_score_items)}):")
            for item in low_score_items:
                print(f"      - Query {item['id']}: {item['query'][:40]}... (分数: {item['score']:.2f})")
    
    return category_stats


def identify_low_faithfulness_cases(results: List[Dict], threshold: float = 0.5) -> List[Dict]:
    """识别低 Faithfulness 案例"""
    print("\n" + "="*80)
    print(f"🔍 低 Faithfulness 案例分析（阈值: {threshold}）")
    print("="*80)
    
    low_cases = []
    
    for r in results:
        f_score = r.get('faithfulness_score')
        if f_score is not None and f_score < threshold:
            low_cases.append({
                'id': r['id'],
                'query': r['query'],
                'category': r.get('category'),
                'faithfulness': f_score,
                'hit': r.get('hit'),
                'answer_rank': r.get('answer_rank'),
                'rerank_top1': r.get('rerank_scores', [None])[0],
                'answer': r.get('answer', ''),
                'judge_response': r.get('judge_response', '')
            })
    
    if not low_cases:
        print(f"\n✅ 没有发现低于 {threshold} 的 Faithfulness 案例！")
        return []
    
    print(f"\n⚠️  发现 {len(low_cases)} 个低 Faithfulness 案例：\n")
    
    for i, case in enumerate(low_cases, 1):
        print(f"{i}. Query {case['id']}: {case['query']}")
        print(f"   Faithfulness: {case['faithfulness']:.2f}")
        print(f"   Category:     {case['category']}")
        print(f"   Hit:          {'✅' if case['hit'] else '❌'}")
        print(f"   Answer Rank:  {case['answer_rank'] if case['answer_rank'] else 'N/A'}")
        print(f"   Rerank Top-1: {case['rerank_top1']:.4f}" if case['rerank_top1'] else "   Rerank Top-1: N/A")
        print(f"\n   答案（前150字）:")
        print(f"   {case['answer'][:150]}...")
        print(f"\n   Judge 评价（前200字）:")
        print(f"   {case['judge_response'][:200]}...")
        print("\n" + "-"*80 + "\n")
    
    return low_cases


def analyze_threshold_for_rejection(results: List[Dict]) -> Dict:
    """
    为 Iteration 6 的拒答机制推荐阈值
    
    分析不同阈值下的拒答率和误拒率
    """
    print("\n" + "="*80)
    print("🎯 Iteration 6 拒答阈值建议")
    print("="*80)
    
    # 提取数据
    faithfulness_scores = []
    hit_flags = []
    
    for r in results:
        f_score = r.get('faithfulness_score')
        if f_score is not None:
            faithfulness_scores.append(f_score)
            hit_flags.append(r.get('hit', False))
    
    if not faithfulness_scores:
        print("\n⚠️  没有 Faithfulness 数据，无法分析阈值")
        return {}
    
    # 测试不同阈值
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"\n假设：Faithfulness < 阈值 → 拒答")
    print(f"{'阈值':<8} {'拒答率':<10} {'误拒率':<10} {'说明'}")
    print("-" * 60)
    
    threshold_analysis = {}
    
    for thresh in thresholds:
        # 统计会被拒答的查询
        rejected_count = sum(1 for score in faithfulness_scores if score < thresh)
        rejection_rate = rejected_count / len(faithfulness_scores)
        
        # 统计被误拒的查询（检索命中但因 Faithfulness 低被拒）
        false_rejection_count = sum(
            1 for score, hit in zip(faithfulness_scores, hit_flags)
            if score < thresh and hit
        )
        false_rejection_rate = false_rejection_count / rejected_count if rejected_count > 0 else 0
        
        if rejection_rate < 0.1:
            level = "保守（几乎不拒答）"
        elif rejection_rate < 0.3:
            level = "推荐（平衡）"
        elif rejection_rate < 0.5:
            level = "中等"
        else:
            level = "激进（拒答多）"
        
        threshold_analysis[thresh] = {
            'rejection_rate': rejection_rate,
            'false_rejection_rate': false_rejection_rate,
            'rejected_count': rejected_count,
            'false_rejection_count': false_rejection_count
        }
        
        print(f"{thresh:<8.2f} {rejection_rate:<10.1%} {false_rejection_rate:<10.1%} {level}")
    
    # 推荐阈值
    print(f"\n💡 推荐策略：")
    print(f"   1. 保守策略（低拒答率）: 0.3-0.4")
    print(f"      适用：初期上线，优先保证用户体验")
    print(f"   2. 平衡策略（推荐）:     0.5-0.6")
    print(f"      适用：正常运营，兼顾质量和体验")
    print(f"   3. 严格策略（高质量）:   0.7-0.8")
    print(f"      适用：对准确性要求极高的场景")
    
    return threshold_analysis


def compare_hit_vs_miss(results: List[Dict]):
    """对比命中和未命中查询的 Faithfulness"""
    print("\n" + "="*80)
    print("🆚 命中 vs 未命中查询的 Faithfulness 对比")
    print("="*80)
    
    hit_scores = []
    miss_scores = []
    
    for r in results:
        f_score = r.get('faithfulness_score')
        if f_score is None:
            continue
        
        if r.get('hit'):
            hit_scores.append(f_score)
        else:
            miss_scores.append(f_score)
    
    if hit_scores:
        print(f"\n✅ 命中查询 (n={len(hit_scores)})")
        print(f"   平均:   {np.mean(hit_scores):.3f}")
        print(f"   中位数: {np.median(hit_scores):.3f}")
        print(f"   范围:   [{np.min(hit_scores):.3f}, {np.max(hit_scores):.3f}]")
    
    if miss_scores:
        print(f"\n❌ 未命中查询 (n={len(miss_scores)})")
        print(f"   平均:   {np.mean(miss_scores):.3f}")
        print(f"   中位数: {np.median(miss_scores):.3f}")
        print(f"   范围:   [{np.min(miss_scores):.3f}, {np.max(miss_scores):.3f}]")
    
    if hit_scores and miss_scores:
        delta = np.mean(hit_scores) - np.mean(miss_scores)
        print(f"\n📊 差异分析:")
        print(f"   平均分差: {delta:+.3f}")
        if delta > 0.2:
            print(f"   💡 命中查询的 Faithfulness 显著更高")
        elif delta < -0.1:
            print(f"   ⚠️  未命中查询的 Faithfulness 反而更高（异常）")
        else:
            print(f"   ⚠️  命中与否对 Faithfulness 影响不大")


def generate_summary_report(results: List[Dict], output_file: str = "faithfulness_analysis_report.md"):
    """生成 Markdown 格式的总结报告"""
    print("\n" + "="*80)
    print("📝 生成总结报告")
    print("="*80)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Iteration 5: Faithfulness 分析报告\n\n")
        
        # 基本统计
        faithfulness_scores = [r.get('faithfulness_score') for r in results if r.get('faithfulness_score') is not None]
        
        f.write("## 📊 整体统计\n\n")
        f.write(f"- **评估查询数**: {len(faithfulness_scores)}\n")
        f.write(f"- **平均 Faithfulness**: {np.mean(faithfulness_scores):.3f}\n")
        f.write(f"- **中位数**: {np.median(faithfulness_scores):.3f}\n")
        f.write(f"- **范围**: [{np.min(faithfulness_scores):.3f}, {np.max(faithfulness_scores):.3f}]\n")
        f.write(f"- **标准差**: {np.std(faithfulness_scores):.3f}\n\n")
        
        # 分数分布
        f.write("## 📈 分数分布\n\n")
        f.write("| 分数区间 | 数量 | 占比 |\n")
        f.write("|---------|------|------|\n")
        
        ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        for low, high in ranges:
            count = sum(1 for s in faithfulness_scores if low <= s < high)
            pct = count / len(faithfulness_scores) * 100 if faithfulness_scores else 0
            f.write(f"| [{low:.1f}, {high:.1f}) | {count} | {pct:.1f}% |\n")
        
        # 低分案例
        low_cases = [r for r in results if r.get('faithfulness_score', 1) < 0.5]
        if low_cases:
            f.write(f"\n## ⚠️ 低分案例 (n={len(low_cases)})\n\n")
            for case in low_cases:
                f.write(f"### Query {case['id']}: {case['query']}\n\n")
                f.write(f"- **Faithfulness**: {case.get('faithfulness_score', 'N/A'):.2f}\n")
                f.write(f"- **Category**: {case.get('category')}\n")
                f.write(f"- **Hit**: {'✅' if case.get('hit') else '❌'}\n")
                f.write(f"- **Answer**:\n  ```\n  {case.get('answer', '')[:200]}...\n  ```\n\n")
        
        f.write("\n## 🎯 Iteration 6 建议\n\n")
        f.write("基于当前数据，建议的拒答阈值：\n\n")
        f.write("- **保守策略**: 0.3 - 0.4 （低拒答率，优先用户体验）\n")
        f.write("- **平衡策略**: 0.5 - 0.6 （推荐，兼顾质量和体验）\n")
        f.write("- **严格策略**: 0.7 - 0.8 （高质量，可能误拒部分正确答案）\n\n")
    
    print(f"✅ 报告已生成: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Iteration 5: Faithfulness 深度分析")
    parser.add_argument("--result-file", default="results_fixed_200_40_rerank.json",
                        help="评估结果 JSON 文件")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="低 Faithfulness 阈值（默认 0.5）")
    parser.add_argument("--output", default="faithfulness_analysis_report.md",
                        help="输出报告文件名")
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔬 Iteration 5: Faithfulness 深度分析")
    print("=" * 80)
    
    # 加载结果
    print(f"\n📂 加载结果文件: {args.result_file}")
    data = load_results(args.result_file)
    results = data.get('results', [])
    
    print(f"✅ 加载 {len(results)} 条查询结果")
    
    # 检查是否有 Faithfulness 数据
    has_faithfulness = any(r.get('faithfulness_score') is not None for r in results)
    if not has_faithfulness:
        print("\n⚠️  警告：未发现 Faithfulness 数据！")
        print("请使用 Judge 模式运行评估：")
        print("  python3 run_eval.py --chunking-strategy fixed_200_40 --retrieval-mode rerank")
        return
    
    # 执行各项分析
    analyze_correlation(results)
    analyze_by_category(results)
    compare_hit_vs_miss(results)
    identify_low_faithfulness_cases(results, threshold=args.threshold)
    analyze_threshold_for_rejection(results)
    
    # 生成报告
    generate_summary_report(results, output_file=args.output)
    
    print("\n" + "=" * 80)
    print("✅ 分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
