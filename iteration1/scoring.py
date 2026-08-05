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
