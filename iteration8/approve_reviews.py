#!/usr/bin/env python3
"""
批量审核通过脚本

使用方法:
    # 通过指定的审核文件
    python approve_reviews.py review_abc123_20260810_120000.json review_def456_20260810_130000.json
    
    # 通过所有待审核文件
    python approve_reviews.py --all
"""

import os
import sys
import argparse
from self_healing import approve_reviews, list_pending_reviews


def main():
    parser = argparse.ArgumentParser(description="Batch approve self-healing reviews")
    parser.add_argument("files", nargs="*", help="Review filenames to approve")
    parser.add_argument("--all", action="store_true", help="Approve all pending reviews")
    parser.add_argument("--review-dir", default="review", help="Review directory")
    parser.add_argument("--corpus-dir", default="corpus", help="Corpus directory")
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print("批量审核通过脚本")
    print(f"{'='*60}\n")
    
    # 确定要审核的文件
    review_files = []
    
    if args.all:
        # 通过所有待审核文件
        pending = list_pending_reviews(args.review_dir)
        review_files = [r['filepath'] for r in pending]
        print(f"找到 {len(review_files)} 个待审核文件")
    elif args.files:
        # 通过指定文件
        for filename in args.files:
            filepath = os.path.join(args.review_dir, filename)
            if os.path.exists(filepath):
                review_files.append(filepath)
            else:
                print(f"⚠️  文件不存在: {filename}")
    else:
        print("❌ 请指定要通过的文件或使用 --all 标志")
        parser.print_help()
        return 1
    
    if not review_files:
        print("❌ 没有文件需要审核")
        return 0
    
    # 确认
    print(f"\n将要通过以下 {len(review_files)} 个审核:")
    for filepath in review_files:
        print(f"  - {os.path.basename(filepath)}")
    
    confirm = input(f"\n确认通过这些审核? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ 已取消")
        return 0
    
    # 执行审核
    print(f"\n{'='*60}")
    print("开始审核...")
    print(f"{'='*60}\n")
    
    result = approve_reviews(review_files, args.corpus_dir, archive=True)
    
    print(f"\n{'='*60}")
    print("审核完成!")
    print(f"{'='*60}")
    print(f"✅ 成功: {result['approved_count']}")
    print(f"⚠️  跳过重复: {result['skipped_count']}")
    print(f"❌ 失败: {result['failed_count']}")
    print(f"📝 创建文档: {len(result['created_docs'])}")
    print(f"📦 归档文件: {result['archived_count']}")
    
    if result['created_docs']:
        print(f"\n新创建的文档:")
        for doc_id in result['created_docs']:
            print(f"  - {doc_id}.txt")
    
    if result['archived_count'] > 0:
        print(f"\n📦 已批准的 review 文件已移动到: {args.review_dir}/approved/")
    
    print(f"\n💡 提示: 需要重新运行评估以查看效果")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
