#!/usr/bin/env python3
"""
清理已批准的 review 文件

将 review/ 目录中所有 status='approved' 的文件移动到 review/approved/
"""

import os
import json
import shutil

def cleanup_approved_reviews(review_dir: str = "review"):
    """移动已批准的 review 文件到归档目录"""
    
    if not os.path.exists(review_dir):
        print(f"❌ Review 目录不存在: {review_dir}")
        return
    
    # 创建归档目录
    approved_dir = os.path.join(review_dir, "approved")
    os.makedirs(approved_dir, exist_ok=True)
    
    moved_count = 0
    pending_count = 0
    
    print(f"{'='*60}")
    print("清理已批准的 Review 文件")
    print(f"{'='*60}\n")
    
    # 遍历 review 目录中的文件
    for filename in os.listdir(review_dir):
        if not filename.endswith('.json') or filename.startswith('.'):
            continue
        
        filepath = os.path.join(review_dir, filename)
        
        # 跳过目录
        if os.path.isdir(filepath):
            continue
        
        try:
            # 读取文件
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            status = data.get('status', 'unknown')
            
            if status == 'approved':
                # 移动到归档目录
                dest_path = os.path.join(approved_dir, filename)
                shutil.move(filepath, dest_path)
                print(f"✅ 移动: {filename}")
                moved_count += 1
            elif status == 'pending':
                pending_count += 1
            else:
                print(f"⚠️  未知状态 '{status}': {filename}")
        
        except Exception as e:
            print(f"❌ 处理失败: {filename} - {e}")
    
    print(f"\n{'='*60}")
    print("清理完成!")
    print(f"{'='*60}")
    print(f"📦 移动到归档: {moved_count} 个文件")
    print(f"📝 保留待审核: {pending_count} 个文件")
    print(f"📁 归档位置: {approved_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    cleanup_approved_reviews()
