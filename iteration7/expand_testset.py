"""
测试集扩充工具

功能：
- 帮助扩充 corpus 和 queries
- 验证新增文档和查询的格式
- 生成文档/查询的统计信息

使用方法：
    # 检查当前测试集统计
    python expand_testset.py --action stats
    
    # 验证新增的文档和查询
    python expand_testset.py --action validate
    
    # 生成新文档模板
    python expand_testset.py --action create-doc --doc-id 8
"""

import os
import json
import argparse
from typing import Dict, List
import re


def get_corpus_stats(corpus_dir: str = "corpus") -> Dict:
    """获取 corpus 统计信息"""
    import glob
    
    doc_files = sorted(glob.glob(os.path.join(corpus_dir, "doc-*.txt")))
    
    stats = {
        "num_documents": len(doc_files),
        "documents": [],
        "total_chars": 0,
        "avg_chars": 0
    }
    
    for doc_file in doc_files:
        doc_id = os.path.basename(doc_file).replace("doc-", "").replace(".txt", "")
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
            char_count = len(content)
            stats["documents"].append({
                "doc_id": doc_id,
                "file": os.path.basename(doc_file),
                "chars": char_count,
                "lines": content.count('\n') + 1
            })
            stats["total_chars"] += char_count
    
    if stats["num_documents"] > 0:
        stats["avg_chars"] = stats["total_chars"] / stats["num_documents"]
    
    return stats


def get_queries_stats(query_file: str = "corpus/queries.json") -> Dict:
    """获取 queries 统计信息"""
    if not os.path.exists(query_file):
        return {"error": f"File not found: {query_file}"}
    
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    stats = {
        "num_queries": len(queries),
        "categories": {},
        "doc_coverage": {},
        "queries": []
    }
    
    for q in queries:
        # 分类统计
        category = q.get("category", "unknown")
        stats["categories"][category] = stats["categories"].get(category, 0) + 1
        
        # 文档覆盖
        doc_id = q.get("doc_id", "unknown")
        stats["doc_coverage"][doc_id] = stats["doc_coverage"].get(doc_id, 0) + 1
        
        # 查询详情
        stats["queries"].append({
            "id": q.get("id"),
            "category": category,
            "doc_id": doc_id,
            "query": q.get("query", "")[:50] + "..." if len(q.get("query", "")) > 50 else q.get("query", "")
        })
    
    return stats


def validate_corpus(corpus_dir: str = "corpus") -> List[str]:
    """验证 corpus 文档格式"""
    issues = []
    import glob
    
    doc_files = glob.glob(os.path.join(corpus_dir, "doc-*.txt"))
    
    # 检查文件名格式
    for doc_file in doc_files:
        basename = os.path.basename(doc_file)
        if not re.match(r'^doc-\d+\.txt$', basename):
            issues.append(f"❌ Invalid filename format: {basename} (expected: doc-N.txt)")
    
    # 检查文件内容
    for doc_file in doc_files:
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            if len(content.strip()) == 0:
                issues.append(f"⚠️  Empty document: {os.path.basename(doc_file)}")
            elif len(content) < 50:
                issues.append(f"⚠️  Very short document ({len(content)} chars): {os.path.basename(doc_file)}")
    
    # 检查文档 ID 连续性
    doc_ids = []
    for doc_file in doc_files:
        match = re.search(r'doc-(\d+)\.txt$', doc_file)
        if match:
            doc_ids.append(int(match.group(1)))
    
    doc_ids.sort()
    for i in range(len(doc_ids) - 1):
        if doc_ids[i+1] - doc_ids[i] > 1:
            issues.append(f"⚠️  Gap in document IDs: {doc_ids[i]} → {doc_ids[i+1]}")
    
    return issues


def validate_queries(query_file: str = "corpus/queries.json", corpus_dir: str = "corpus") -> List[str]:
    """验证 queries 格式"""
    issues = []
    
    if not os.path.exists(query_file):
        issues.append(f"❌ Query file not found: {query_file}")
        return issues
    
    with open(query_file, 'r', encoding='utf-8') as f:
        try:
            queries = json.load(f)
        except json.JSONDecodeError as e:
            issues.append(f"❌ Invalid JSON format: {e}")
            return issues
    
    # 检查必需字段
    required_fields = ["id", "query", "doc_id", "char_start", "char_end", "category"]
    
    for i, q in enumerate(queries):
        for field in required_fields:
            if field not in q:
                issues.append(f"❌ Query {i}: Missing field '{field}'")
        
        # 检查 query 是否为空
        if "query" in q and len(q["query"].strip()) == 0:
            issues.append(f"⚠️  Query {q.get('id', i)}: Empty query text")
        
        # 检查 doc_id 是否存在对应文档
        if "doc_id" in q:
            doc_file = os.path.join(corpus_dir, f"doc-{q['doc_id']}.txt")
            if not os.path.exists(doc_file):
                issues.append(f"❌ Query {q.get('id', i)}: Document not found (doc-{q['doc_id']}.txt)")
            else:
                # 检查 char_start/char_end 是否合法
                with open(doc_file, 'r', encoding='utf-8') as df:
                    doc_content = df.read()
                    char_start = q.get("char_start", 0)
                    char_end = q.get("char_end", 0)
                    
                    if char_start < 0 or char_end > len(doc_content):
                        issues.append(f"⚠️  Query {q.get('id', i)}: char_start/char_end out of range (doc length: {len(doc_content)})")
                    
                    if char_start >= char_end:
                        issues.append(f"⚠️  Query {q.get('id', i)}: char_start >= char_end")
    
    # 检查 ID 唯一性
    ids = [q.get("id") for q in queries]
    duplicates = [id for id in ids if ids.count(id) > 1]
    if duplicates:
        issues.append(f"❌ Duplicate query IDs: {set(duplicates)}")
    
    return issues


def create_doc_template(doc_id: int, corpus_dir: str = "corpus") -> str:
    """创建新文档模板"""
    doc_file = os.path.join(corpus_dir, f"doc-{doc_id}.txt")
    
    if os.path.exists(doc_file):
        return f"⚠️  Document already exists: {doc_file}"
    
    template = f"""# Document {doc_id}

[在这里填写文档内容]

提示：
1. 文档内容应该包含可以回答问题的信息
2. 建议字数：200-1000 字
3. 可以是产品说明、技术文档、FAQ 等
4. 确保内容准确、完整

示例主题建议：
- 产品功能介绍
- 使用教程
- 故障排除
- 技术规格
- 常见问题解答
"""
    
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write(template)
    
    return f"✅ Template created: {doc_file}"


def create_query_template(corpus_dir: str = "corpus") -> str:
    """生成查询模板（在现有 queries.json 末尾）"""
    query_file = os.path.join(corpus_dir, "queries.json")
    
    if not os.path.exists(query_file):
        return f"❌ Query file not found: {query_file}"
    
    with open(query_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    # 获取下一个 ID
    max_id = max([q.get("id", 0) for q in queries]) if queries else 0
    next_id = max_id + 1
    
    template = {
        "id": next_id,
        "query": "[在这里填写查询问题]",
        "doc_id": "X",
        "char_start": 0,
        "char_end": 100,
        "category": "factual|reasoning|multi-hop",
        "_note": "请根据实际情况修改 doc_id, char_start, char_end 和 category"
    }
    
    print(f"\n📝 New Query Template (ID: {next_id}):")
    print(json.dumps(template, ensure_ascii=False, indent=2))
    print(f"\n💡 提示：")
    print(f"   1. 复制上面的 JSON，添加到 {query_file} 的数组末尾")
    print(f"   2. 修改 doc_id 为目标文档 ID")
    print(f"   3. 在文档中找到答案位置，更新 char_start 和 char_end")
    print(f"   4. 选择合适的 category: factual, reasoning, multi-hop")
    print(f"   5. 删除 _note 字段")
    
    return ""


def print_stats():
    """打印统计信息"""
    print(f"\n{'='*60}")
    print("📊 Corpus & Queries Statistics")
    print(f"{'='*60}\n")
    
    # Corpus 统计
    corpus_stats = get_corpus_stats()
    print(f"📁 Documents: {corpus_stats['num_documents']}")
    print(f"   Total chars: {corpus_stats['total_chars']:,}")
    print(f"   Average: {corpus_stats['avg_chars']:.0f} chars/doc\n")
    
    print("Documents:")
    for doc in corpus_stats['documents']:
        print(f"   • doc-{doc['doc_id']}.txt: {doc['chars']:,} chars, {doc['lines']} lines")
    
    # Queries 统计
    print()
    queries_stats = get_queries_stats()
    if "error" in queries_stats:
        print(f"❌ {queries_stats['error']}")
    else:
        print(f"❓ Queries: {queries_stats['num_queries']}")
        print(f"\n   By Category:")
        for cat, count in queries_stats['categories'].items():
            print(f"      • {cat}: {count}")
        
        print(f"\n   By Document:")
        for doc_id, count in sorted(queries_stats['doc_coverage'].items()):
            print(f"      • doc-{doc_id}: {count} queries")
        
        # 检查覆盖率
        print(f"\n   Coverage:")
        covered_docs = len(queries_stats['doc_coverage'])
        total_docs = corpus_stats['num_documents']
        coverage = covered_docs / total_docs if total_docs > 0 else 0
        print(f"      • Documents covered: {covered_docs}/{total_docs} ({coverage:.0%})")
        
        if coverage < 1.0:
            uncovered = set(str(d['doc_id']) for d in corpus_stats['documents']) - set(queries_stats['doc_coverage'].keys())
            print(f"      • Uncovered: doc-{', doc-'.join(sorted(uncovered))}")
    
    print(f"\n{'='*60}")


def print_validation():
    """打印验证结果"""
    print(f"\n{'='*60}")
    print("🔍 Validation Results")
    print(f"{'='*60}\n")
    
    # 验证 corpus
    print("📁 Corpus Validation:")
    corpus_issues = validate_corpus()
    if not corpus_issues:
        print("   ✅ All documents are valid")
    else:
        for issue in corpus_issues:
            print(f"   {issue}")
    
    # 验证 queries
    print("\n❓ Queries Validation:")
    query_issues = validate_queries()
    if not query_issues:
        print("   ✅ All queries are valid")
    else:
        for issue in query_issues:
            print(f"   {issue}")
    
    print(f"\n{'='*60}")
    
    if not corpus_issues and not query_issues:
        print("✅ All validations passed!")
    else:
        print(f"⚠️  Found {len(corpus_issues) + len(query_issues)} issues")
    print(f"{'='*60}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试集扩充工具")
    parser.add_argument("--action", required=True, 
                       choices=["stats", "validate", "create-doc", "create-query"],
                       help="操作类型")
    parser.add_argument("--doc-id", type=int, help="新文档 ID（用于 create-doc）")
    parser.add_argument("--corpus-dir", default="corpus", help="Corpus 目录")
    
    args = parser.parse_args()
    
    if args.action == "stats":
        print_stats()
    
    elif args.action == "validate":
        print_validation()
    
    elif args.action == "create-doc":
        if not args.doc_id:
            print("❌ Error: --doc-id is required for create-doc action")
            return
        result = create_doc_template(args.doc_id, args.corpus_dir)
        print(result)
    
    elif args.action == "create-query":
        result = create_query_template(args.corpus_dir)
        if result:
            print(result)


if __name__ == "__main__":
    main()
