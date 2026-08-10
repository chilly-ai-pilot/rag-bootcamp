#!/usr/bin/env python3
"""
自动更新 queries.json 中的 doc_id 映射

当新的 Q&A 文档被添加到 corpus/ 后，这个脚本会：
1. 扫描所有 Q&A 格式的文档（问：...答：...）
2. 匹配 queries.json 中的查询问题
3. 更新对应的 doc_id、char_start、char_end 和 ground_truth_text

用途：在 approve reviews 后自动执行，确保 ground truth 指向最新的文档
"""

import json
import os
import re
from typing import Dict, List, Tuple


def extract_qa_info(doc_path: str, doc_id: str) -> Dict:
    """
    从 Q&A 文档中提取问题和答案信息
    
    参数:
        doc_path: 文档文件路径
        doc_id: 文档 ID（如 "doc17"）
    
    返回:
        包含 question, answer, char_start, char_end 的字典
        如果不是 Q&A 格式，返回 None
    """
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否是 Q&A 格式
        if not content.startswith('问：'):
            return None
        
        # 提取问题
        lines = content.split('\n')
        question = lines[0].replace('问：', '').strip()
        
        # 提取答案
        answer_line = None
        for line in lines:
            if line.startswith('答：'):
                answer_line = line
                break
        
        if not answer_line:
            return None
        
        # 计算答案在文档中的位置
        answer_start = content.find('答：') + 2  # "答："之后
        answer_end = len(content.strip())
        answer_text = content[answer_start:answer_end].strip()
        
        return {
            'doc_id': doc_id,
            'question': question,
            'answer': answer_text,
            'char_start': answer_start,
            'char_end': answer_end
        }
    
    except Exception as e:
        print(f"⚠️  读取 {doc_path} 失败: {e}")
        return None


def normalize_question(question: str) -> str:
    """
    标准化问题文本，用于模糊匹配
    
    去除标点符号、空格，转为小写
    """
    # 去除常见标点
    question = re.sub(r'[？?！!。，,、；;：:""'']', '', question)
    # 去除空格
    question = question.replace(' ', '')
    return question.lower()


def find_qa_documents(corpus_dir: str) -> Dict[str, Dict]:
    """
    扫描 corpus 目录，找到所有 Q&A 文档
    
    参数:
        corpus_dir: 语料库目录路径
    
    返回:
        字典 {doc_id: qa_info}
    """
    qa_docs = {}
    
    # 遍历所有 .txt 文件
    for fname in sorted(os.listdir(corpus_dir)):
        if not fname.endswith('.txt'):
            continue
        
        # 提取文档 ID: doc-17.txt -> doc17
        doc_id = 'doc' + fname.replace('doc-', '').replace('.txt', '')
        doc_path = os.path.join(corpus_dir, fname)
        
        # 提取 Q&A 信息
        qa_info = extract_qa_info(doc_path, doc_id)
        
        if qa_info:
            qa_docs[doc_id] = qa_info
    
    return qa_docs


def match_queries_to_qa_docs(queries: List[Dict], qa_docs: Dict[str, Dict]) -> List[Tuple]:
    """
    匹配 queries 和 Q&A 文档
    
    参数:
        queries: queries.json 中的查询列表
        qa_docs: Q&A 文档信息字典
    
    返回:
        匹配列表: [(query_id, old_doc_id, new_doc_id, qa_info), ...]
    
    注意：不在这里做去重检查，去重逻辑在 approve_reviews.py 中处理
    """
    matches = []
    
    for query in queries:
        query_id = query['id']
        query_text = query['query']
        old_doc_id = query['doc_id']
        
        # 标准化查询文本
        query_norm = normalize_question(query_text)
        
        # 尝试匹配 Q&A 文档
        for doc_id, qa_info in qa_docs.items():
            qa_question = qa_info['question']
            qa_norm = normalize_question(qa_question)
            
            # 模糊匹配：任一包含另一个
            if qa_norm in query_norm or query_norm in qa_norm:
                matches.append({
                    'query_id': query_id,
                    'old_doc_id': old_doc_id,
                    'new_doc_id': doc_id,
                    'qa_info': qa_info
                })
                break
    
    return matches


def update_queries_json(
    query_file: str,
    corpus_dir: str,
    dry_run: bool = False,
    verbose: bool = True
) -> int:
    """
    更新 queries.json 中的 doc_id 映射
    
    参数:
        query_file: queries.json 文件路径
        corpus_dir: 语料库目录路径
        dry_run: 如果为 True，只打印变更但不实际修改文件
        verbose: 是否打印详细信息
    
    返回:
        更新的查询数量
    """
    # 1. 加载 queries.json
    if not os.path.exists(query_file):
        print(f"❌ 文件不存在: {query_file}")
        return 0
    
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    if verbose:
        print(f"✅ 加载了 {len(queries)} 个查询")
    
    # 2. 扫描 Q&A 文档
    qa_docs = find_qa_documents(corpus_dir)
    
    if verbose:
        print(f"✅ 找到 {len(qa_docs)} 个 Q&A 文档")
        if qa_docs:
            print(f"   Q&A 文档: {sorted(qa_docs.keys(), key=lambda x: int(x.replace('doc', '')))}")
    
    if not qa_docs:
        if verbose:
            print("ℹ️  没有找到 Q&A 文档，无需更新")
        return 0
    
    # 3. 匹配查询和 Q&A 文档
    matches = match_queries_to_qa_docs(queries, qa_docs)
    
    if not matches:
        if verbose:
            print("ℹ️  没有匹配的查询，无需更新")
        return 0
    
    if verbose:
        print(f"\n🔍 找到 {len(matches)} 个匹配:\n")
    
    # 4. 更新 queries
    updated_count = 0
    
    for match in matches:
        query_id = match['query_id']
        old_doc_id = match['old_doc_id']
        new_doc_id = match['new_doc_id']
        qa_info = match['qa_info']
        
        # 找到对应的 query
        query = next((q for q in queries if q['id'] == query_id), None)
        
        if not query:
            continue
        
        # 检查是否需要更新
        needs_update = (
            query['doc_id'] != new_doc_id or
            query.get('char_start') != qa_info['char_start'] or
            query.get('char_end') != qa_info['char_end'] or
            query.get('ground_truth_text') != qa_info['answer']
        )
        
        if not needs_update:
            if verbose:
                print(f"⏭️  Query #{query_id}: 已经是最新 ({new_doc_id})")
            continue
        
        # 安全检查：确保新文档确实存在
        doc_num = new_doc_id.replace('doc', '')
        new_doc_path = os.path.join(corpus_dir, f"doc-{doc_num}.txt")
        if not os.path.exists(new_doc_path):
            if verbose:
                print(f"⚠️  Query #{query_id}: 跳过更新 ({old_doc_id} -> {new_doc_id})")
                print(f"   原因: 目标文档不存在（可能因重复被跳过）\n")
            continue
        
        if verbose:
            print(f"✏️  Query #{query_id}: {old_doc_id} -> {new_doc_id}")
            print(f"   Query: {query['query']}")
            print(f"   Ground Truth: {qa_info['answer'][:60]}...")
            print(f"   Position: {qa_info['char_start']}-{qa_info['char_end']}\n")
        
        # 更新字段
        if not dry_run:
            query['doc_id'] = new_doc_id
            query['char_start'] = qa_info['char_start']
            query['char_end'] = qa_info['char_end']
            query['ground_truth_text'] = qa_info['answer']
        
        updated_count += 1
    
    # 5. 保存更新后的 queries.json
    if updated_count > 0 and not dry_run:
        with open(query_file, 'w', encoding='utf-8') as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
        
        if verbose:
            print(f"\n✅ 已更新 {updated_count} 个查询，保存到 {query_file}")
    elif updated_count > 0 and dry_run:
        if verbose:
            print(f"\n🔍 DRY RUN: 发现 {updated_count} 个查询需要更新（未实际修改）")
    else:
        if verbose:
            print("\nℹ️  所有查询都是最新的，无需更新")
    
    return updated_count


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='自动更新 queries.json 中的 doc_id 映射',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 更新 queries.json
  python update_queries_docid.py
  
  # 指定路径
  python update_queries_docid.py --query-file corpus/queries.json --corpus-dir corpus
  
  # 试运行（不实际修改文件）
  python update_queries_docid.py --dry-run
  
  # 静默模式（只输出最终结果）
  python update_queries_docid.py --quiet
        """
    )
    
    parser.add_argument(
        '--query-file',
        default='corpus/queries.json',
        help='queries.json 文件路径（默认: corpus/queries.json）'
    )
    
    parser.add_argument(
        '--corpus-dir',
        default='corpus',
        help='语料库目录路径（默认: corpus）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='试运行：只打印变更但不实际修改文件'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='静默模式：只输出最终结果'
    )
    
    args = parser.parse_args()
    
    # 执行更新
    updated_count = update_queries_json(
        query_file=args.query_file,
        corpus_dir=args.corpus_dir,
        dry_run=args.dry_run,
        verbose=not args.quiet
    )
    
    # 返回状态码
    if updated_count > 0:
        exit(0)  # 有更新
    else:
        exit(0)  # 无更新但也不是错误


if __name__ == '__main__':
    main()
