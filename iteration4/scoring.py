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
