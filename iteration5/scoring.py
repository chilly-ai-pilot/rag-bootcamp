"""
评分模块

Iteration 0 只需要 Recall@K（检索命中/未命中）指标，
以及一个能查看生成输出的地方——忠实度评分是 Iteration 5 的内容
（LLM-as-judge / Ragas），不在这里。暂时不要构建那部分。
"""
from collections import defaultdict
from typing import List, Dict


def hit(retrieved_chunks: List[Dict], gt_doc_id: str, gt_start: int, gt_end: int) -> int:
    """判断检索结果是否命中真实答案
    
    检查检索到的块中是否有任何一个与正确文档中的真实答案区间重叠
    
    参数:
        retrieved_chunks: 检索到的文档块列表
        gt_doc_id: 真实答案所在的文档 ID（ground truth）
        gt_start: 真实答案在文档中的起始字符位置
        gt_end: 真实答案在文档中的结束字符位置
    
    返回:
        1 表示命中，0 表示未命中
    """
    for c in retrieved_chunks:
        # 检查三个条件：
        # 1. 文档 ID 匹配
        # 2. 块的起始位置在答案结束前（c["start"] < gt_end）
        # 3. 块的结束位置在答案开始后（c["end"] > gt_start）
        # 满足这三个条件说明有区间重叠
        if c["doc_id"] == gt_doc_id and c["start"] < gt_end and c["end"] > gt_start:
            return 1
    return 0


def calculate_mrr(results: List[Dict]) -> Dict[str, float]:
    """计算 MRR (Mean Reciprocal Rank) - 平均倒数排名
    
    MRR 衡量答案块在检索结果中的排名质量：
    - 答案排第1名：1/1 = 1.0
    - 答案排第3名：1/3 = 0.33
    - 答案排第5名：1/5 = 0.20
    - 未命中（不在top-k）：0.0
    
    MRR 越高，说明答案越靠前，生成器越容易"看到"正确答案。
    
    参数:
        results: 结果列表，每项包含 {
            'category': str,
            'hit': 0|1,
            'answer_rank': int (1-based, 如果hit=0则为None)
        }
    
    返回:
        每个类别的 MRR 字典，包括 overall（总体MRR）
    """
    # 按类别分桶收集 reciprocal rank 值
    buckets = defaultdict(list)
    for r in results:
        if r['hit'] == 1 and 'answer_rank' in r and r['answer_rank'] is not None:
            # 命中且有排名：计算倒数排名
            rr = 1.0 / r['answer_rank']
        else:
            # 未命中或无排名：倒数排名为0
            rr = 0.0
        buckets[r["category"]].append(rr)
    
    # 计算每个类别的平均倒数排名
    mrr_scores = {cat: sum(v) / len(v) for cat, v in buckets.items()}
    
    # 计算总体 MRR
    all_rr = []
    for r in results:
        if r['hit'] == 1 and 'answer_rank' in r and r['answer_rank'] is not None:
            all_rr.append(1.0 / r['answer_rank'])
        else:
            all_rr.append(0.0)
    mrr_scores["overall"] = sum(all_rr) / len(all_rr) if all_rr else 0.0
    
    return mrr_scores


def find_answer_rank(retrieved_chunks: List[Dict], gt_doc_id: str, gt_start: int, gt_end: int) -> int:
    """找到答案块在检索结果中的排名（1-based）
    
    参数:
        retrieved_chunks: 检索到的文档块列表（按检索器排序）
        gt_doc_id: 真实答案所在的文档 ID
        gt_start: 真实答案在文档中的起始字符位置
        gt_end: 真实答案在文档中的结束字符位置
    
    返回:
        答案块的排名（1-based）。如果未找到，返回 None
    """
    for rank, c in enumerate(retrieved_chunks, start=1):
        # 检查是否与答案区间重叠
        if c["doc_id"] == gt_doc_id and c["start"] < gt_end and c["end"] > gt_start:
            return rank
    return None


def aggregate_by_category(results: List[Dict]) -> Dict[str, float]:
    """按类别聚合计算 Recall@K
    
    参数:
        results: 结果列表，每项包含 {..., 'category': str, 'hit': 0|1}
    
    返回:
        每个类别的 Recall@K 字典，包括 overall（总体召回率）
    """
    # 按类别分桶收集 hit 值
    buckets = defaultdict(list)
    for r in results:
        buckets[r["category"]].append(r["hit"])
    
    # 计算每个类别的召回率（命中数 / 查询总数）
    scores = {cat: sum(v) / len(v) for cat, v in buckets.items()}
    
    # 计算总体召回率
    scores["overall"] = sum(r["hit"] for r in results) / len(results) if results else 0.0
    
    return scores



def analyze_rerank_score_distribution(results: List[Dict]) -> Dict:
    """分析 rerank 分数分布（为 Iteration 6 拒答阈值设计做准备）
    
    收集所有 query 的 rerank 分数，统计分布特征：
    - 最小值、最大值、平均值、中位数
    - 按命中/未命中分组的分数分布
    - 推荐的拒答阈值
    
    这些数据将用于 Iteration 6 设计拒答机制的阈值。
    
    参数:
        results: 评估结果列表，每项包含 {
            'hit': 0|1,
            'rerank_scores': [float, ...] (top-5 的 rerank 分数)
        }
    
    返回:
        分数分布统计字典，包含：
        - all_scores: 所有分数列表
        - hit_scores: 命中 query 的最高分数列表
        - miss_scores: 未命中 query 的最高分数列表
        - statistics: 统计数据（min, max, mean, median, percentiles）
        - threshold_suggestion: 建议的拒答阈值
    """
    all_scores = []
    hit_top_scores = []  # 命中 query 的最高分
    miss_top_scores = []  # 未命中 query 的最高分
    
    for r in results:
        if 'rerank_scores' not in r or not r['rerank_scores']:
            continue
        
        scores = r['rerank_scores']
        all_scores.extend(scores)
        
        # 记录该 query 的最高分（top-1 分数）
        top_score = max(scores)
        
        if r['hit'] == 1:
            hit_top_scores.append(top_score)
        else:
            miss_top_scores.append(top_score)
    
    if not all_scores:
        return {
            "error": "No rerank scores found in results. Make sure to run with --retrieval-mode rerank"
        }
    
    # 计算统计数据
    all_scores_sorted = sorted(all_scores)
    n = len(all_scores_sorted)
    
    statistics = {
        "total_scores": n,
        "min": min(all_scores),
        "max": max(all_scores),
        "mean": sum(all_scores) / n,
        "median": all_scores_sorted[n // 2],
        "p25": all_scores_sorted[n // 4],
        "p75": all_scores_sorted[3 * n // 4],
        "p90": all_scores_sorted[int(0.9 * n)],
        "p95": all_scores_sorted[int(0.95 * n)],
    }
    
    # 分组统计
    hit_stats = {}
    if hit_top_scores:
        hit_sorted = sorted(hit_top_scores)
        hit_stats = {
            "count": len(hit_top_scores),
            "min": min(hit_top_scores),
            "max": max(hit_top_scores),
            "mean": sum(hit_top_scores) / len(hit_top_scores),
            "median": hit_sorted[len(hit_sorted) // 2],
        }
    
    miss_stats = {}
    if miss_top_scores:
        miss_sorted = sorted(miss_top_scores)
        miss_stats = {
            "count": len(miss_top_scores),
            "min": min(miss_top_scores),
            "max": max(miss_top_scores),
            "mean": sum(miss_top_scores) / len(miss_top_scores),
            "median": miss_sorted[len(miss_sorted) // 2],
        }
    
    # 推荐阈值：基于命中和未命中的分数分布
    threshold_suggestion = None
    if hit_top_scores and miss_top_scores:
        # 建议阈值：命中分数的 10th percentile 和未命中分数的 90th percentile 之间
        hit_p10 = sorted(hit_top_scores)[int(0.1 * len(hit_top_scores))]
        miss_p90 = sorted(miss_top_scores)[int(0.9 * len(miss_top_scores))]
        threshold_suggestion = {
            "conservative": hit_p10,  # 保守阈值：只拒绝明显低分的
            "aggressive": miss_p90,   # 激进阈值：尽量避免错误答案
            "recommended": (hit_p10 + miss_p90) / 2,  # 推荐：中间值
            "explanation": f"命中query的P10={hit_p10:.4f}, 未命中query的P90={miss_p90:.4f}"
        }
    
    return {
        "all_scores": all_scores,
        "hit_top_scores": hit_top_scores,
        "miss_top_scores": miss_top_scores,
        "statistics": statistics,
        "hit_statistics": hit_stats,
        "miss_statistics": miss_stats,
        "threshold_suggestion": threshold_suggestion
    }
