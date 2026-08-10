"""
Self-Healing Module for RAG System

自动发现知识库缺陷并生成审核文件，等待人工批准后补充到 corpus。
"""

import os
import json
import hashlib
from typing import Dict, List, Optional
from datetime import datetime


def generate_query_hash(query: str) -> str:
    """生成查询的哈希值（用于去重）"""
    return hashlib.md5(query.encode('utf-8')).hexdigest()[:8]


def should_trigger_self_healing(
    result: Dict,
    self_healing_config: Dict
) -> tuple[bool, str]:
    """
    判断是否应该触发自愈机制
    
    参数:
        result: 评估结果
        self_healing_config: 自愈配置
    
    返回:
        (should_trigger, reason)
    """
    if not self_healing_config.get('enabled', False):
        return False, None
    
    triggers = self_healing_config.get('triggers', {})
    
    # 触发条件 1: hit != 1（未命中）
    if triggers.get('hit_not_1', False):
        if result.get('hit', 0) != 1:
            return True, "retrieval_miss"
    
    # 触发条件 2: answer_rank > threshold
    rank_threshold = triggers.get('answer_rank_threshold', 4)
    answer_rank = result.get('answer_rank')
    if answer_rank is not None and answer_rank > rank_threshold:
        return True, f"low_rank_{answer_rank}"
    
    # 触发条件 3: layer1_rejection (low_score_rejection)
    if triggers.get('layer1_rejection', False):
        if result.get('rejected', False):
            rejection_reason = result.get('rejection_reason', '')
            if 'Layer 1' in rejection_reason or 'layer1' in rejection_reason.lower():
                return True, "low_score_rejection"
    
    return False, None


def create_review_file(
    query: str,
    ground_truth: str,
    doc_id: str,
    char_start: int,
    char_end: int,
    trigger_reason: str,
    rejection_reason: Optional[str],
    review_dir: str = "review"
) -> str:
    """
    创建审核文件
    
    参数:
        query: 查询问题
        ground_truth: 真实答案
        doc_id: 文档 ID
        char_start: 答案起始位置
        char_end: 答案结束位置
        trigger_reason: 触发原因
        rejection_reason: 拒答原因（如果有）
        review_dir: 审核目录
    
    返回:
        创建的文件路径
    """
    # 确保 review 目录存在
    os.makedirs(review_dir, exist_ok=True)
    
    # 生成文件名（基于 query 哈希）
    query_hash = generate_query_hash(query)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"review_{query_hash}_{timestamp}.json"
    filepath = os.path.join(review_dir, filename)
    
    # 创建审核数据
    review_data = {
        "query": query,
        "ground_truth": ground_truth,
        "source": {
            "doc_id": doc_id,
            "char_start": char_start,
            "char_end": char_end
        },
        "trigger_reason": trigger_reason,
        "rejection_reason": rejection_reason,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "query_hash": query_hash
    }
    
    # 写入文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(review_data, f, ensure_ascii=False, indent=2)
    
    return filepath


def deduplicate_review_files(review_dir: str = "review") -> int:
    """
    去重审核文件（根据 query_hash）
    
    保留最新的文件，删除旧的重复文件。
    
    返回:
        删除的文件数量
    """
    if not os.path.exists(review_dir):
        return 0
    
    # 收集所有审核文件
    review_files = {}
    for filename in os.listdir(review_dir):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(review_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            query_hash = data.get('query_hash')
            created_at = data.get('created_at', '')
            
            if query_hash:
                if query_hash not in review_files:
                    review_files[query_hash] = []
                review_files[query_hash].append((filepath, created_at))
        except:
            continue
    
    # 对每个 query_hash，保留最新的文件
    deleted_count = 0
    for query_hash, files in review_files.items():
        if len(files) <= 1:
            continue
        
        # 按时间排序，保留最新的
        files.sort(key=lambda x: x[1], reverse=True)
        
        # 删除旧的
        for filepath, _ in files[1:]:
            try:
                os.remove(filepath)
                deleted_count += 1
                print(f"  删除重复文件: {os.path.basename(filepath)}")
            except:
                pass
    
    return deleted_count


def process_self_healing(
    results: List[Dict],
    queries: List[Dict],
    self_healing_config: Dict
) -> Dict:
    """
    处理自愈逻辑
    
    参数:
        results: 评估结果列表
        queries: 查询列表（包含 ground_truth）
        self_healing_config: 自愈配置
    
    返回:
        {
            "triggered_count": int,
            "created_files": List[str],
            "deduplicated_count": int
        }
    """
    if not self_healing_config.get('enabled', False):
        return {
            "triggered_count": 0,
            "created_files": [],
            "deduplicated_count": 0
        }
    
    review_dir = self_healing_config.get('review_dir', 'review')
    auto_deduplicate = self_healing_config.get('auto_deduplicate', True)
    
    # 创建 query_id -> query_data 映射
    query_map = {q['id']: q for q in queries}
    
    triggered_count = 0
    created_files = []
    
    print(f"\n{'='*60}")
    print("自愈机制检查")
    print(f"{'='*60}")
    
    for result in results:
        query_id = result.get('id')
        if query_id not in query_map:
            continue
        
        query_data = query_map[query_id]
        
        # 检查是否触发自愈
        should_trigger, trigger_reason = should_trigger_self_healing(result, self_healing_config)
        
        if should_trigger:
            triggered_count += 1
            
            # 创建审核文件
            filepath = create_review_file(
                query=query_data['query'],
                ground_truth=query_data.get('ground_truth_text', ''),
                doc_id=query_data['doc_id'],
                char_start=query_data['char_start'],
                char_end=query_data['char_end'],
                trigger_reason=trigger_reason,
                rejection_reason=result.get('rejection_reason'),
                review_dir=review_dir
            )
            
            created_files.append(filepath)
            
            print(f"  ⚠️  触发自愈: Q{query_id}")
            print(f"     原因: {trigger_reason}")
            print(f"     文件: {os.path.basename(filepath)}")
    
    # 去重
    deduplicated_count = 0
    if auto_deduplicate and created_files:
        print(f"\n去重审核文件...")
        deduplicated_count = deduplicate_review_files(review_dir)
    
    if triggered_count > 0:
        print(f"\n✅ 自愈检查完成:")
        print(f"   触发次数: {triggered_count}")
        print(f"   创建文件: {len(created_files)}")
        print(f"   去重删除: {deduplicated_count}")
        print(f"   待审核: {len(created_files) - deduplicated_count}")
    else:
        print(f"✅ 无需触发自愈")
    
    print(f"{'='*60}\n")
    
    return {
        "triggered_count": triggered_count,
        "created_files": created_files,
        "deduplicated_count": deduplicated_count
    }


def list_pending_reviews(review_dir: str = "review") -> List[Dict]:
    """
    列出所有待审核的文件
    
    返回:
        待审核文件列表
    """
    if not os.path.exists(review_dir):
        return []
    
    pending_reviews = []
    
    for filename in os.listdir(review_dir):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(review_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get('status') == 'pending':
                data['filename'] = filename
                data['filepath'] = filepath
                pending_reviews.append(data)
        except:
            continue
    
    # 按创建时间倒序排序
    pending_reviews.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return pending_reviews


def approve_reviews(
    review_files: List[str],
    corpus_dir: str = "corpus"
) -> Dict:
    """
    批量通过审核，将内容添加到 corpus
    
    参数:
        review_files: 要通过的审核文件路径列表
        corpus_dir: corpus 目录
    
    返回:
        {
            "approved_count": int,
            "failed_count": int,
            "created_docs": List[str]
        }
    """
    approved_count = 0
    failed_count = 0
    created_docs = []
    
    # 找到现有文档的最大编号
    existing_docs = []
    for fname in os.listdir(corpus_dir):
        if fname.startswith('doc-') and fname.endswith('.txt'):
            try:
                doc_num = int(fname.replace('doc-', '').replace('.txt', ''))
                existing_docs.append(doc_num)
            except:
                continue
    
    next_doc_num = max(existing_docs) + 1 if existing_docs else 1
    
    for filepath in review_files:
        try:
            # 读取审核文件
            with open(filepath, 'r', encoding='utf-8') as f:
                review_data = json.load(f)
            
            # 提取信息
            ground_truth = review_data['ground_truth']
            query = review_data['query']
            
            # 创建新文档
            new_doc_id = f"doc-{next_doc_num}"
            doc_file = os.path.join(corpus_dir, f"{new_doc_id}.txt")
            
            # 写入内容（包含问题作为标题）
            with open(doc_file, 'w', encoding='utf-8') as f:
                f.write(f"# 问题: {query}\n\n")
                f.write(ground_truth)
            
            created_docs.append(new_doc_id)
            next_doc_num += 1
            
            # 更新审核状态
            review_data['status'] = 'approved'
            review_data['approved_at'] = datetime.now().isoformat()
            review_data['new_doc_id'] = new_doc_id
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(review_data, f, ensure_ascii=False, indent=2)
            
            approved_count += 1
            
        except Exception as e:
            print(f"  ❌ 审核失败: {filepath} - {e}")
            failed_count += 1
    
    return {
        "approved_count": approved_count,
        "failed_count": failed_count,
        "created_docs": created_docs
    }


# 用于测试
if __name__ == "__main__":
    # 测试去重
    print("测试审核文件去重...")
    count = deduplicate_review_files("review")
    print(f"删除了 {count} 个重复文件")
    
    # 列出待审核
    print("\n待审核文件:")
    pending = list_pending_reviews("review")
    for review in pending:
        print(f"  - {review['filename']}")
        print(f"    Query: {review['query']}")
        print(f"    Reason: {review['trigger_reason']}")
