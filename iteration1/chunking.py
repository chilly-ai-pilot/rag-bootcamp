"""
Iteration 0/1 基线分块策略模块

分块大小默认为 200（而非 500）是因为实际语料库文档大小为 355~973 字符
（通过 len() 实测，不是估算）。如果使用 500/块，doc5/doc6/doc7 会各自坍缩
成单个块，那就失去了比较分块策略的意义。详细字符计数历史见 /areas 笔记。

Iteration 2 将在此添加语义边界和更小滑动窗口的变体函数，保持相同的返回格式，
这样 run_eval.py 就可以无需修改其他代码就能切换策略。
"""
import os
from typing import List, Dict


def fixed_length_chunks(text: str, doc_id: str, chunk_size: int = 200, overlap: int = 40) -> List[Dict]:
    """将文本分割成带重叠的固定长度窗口
    
    参数:
        text: 待分块的原始文本
        doc_id: 文档标识符（如 'doc1'）
        chunk_size: 每个块的字符数，默认 200
        overlap: 相邻块之间的重叠字符数，默认 40
    
    返回:
        字典列表: [{doc_id, start, end, text}, ...]
        其中 start/end 是相对于原始文档的字符偏移量，可以直接与
        queries.json 中的 char_start/char_end 进行比较
    """
    # 检查参数有效性：重叠区不能大于等于块大小
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    n = len(text)
    start = 0
    
    # 滑动窗口切分文本
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append({"doc_id": doc_id, "start": start, "end": end, "text": text[start:end]})
        
        # 如果已到达文本末尾，停止分块
        if end == n:
            break
        
        # 下一个块的起始位置：当前结束位置减去重叠区
        start = end - overlap
    
    return chunks


def load_corpus(corpus_dir: str) -> Dict[str, str]:
    """加载语料库中的所有文档
    
    参数:
        corpus_dir: 语料库目录路径
    
    返回:
        字典 {doc_id: full_text}，doc_id 与 queries.json 中的格式匹配（如 'doc1'）
    """
    docs = {}
    
    # 遍历目录下的所有 .txt 文件
    for fname in sorted(os.listdir(corpus_dir)):
        if not fname.endswith(".txt"):
            continue
        
        # 文件名转换: doc-1.txt -> doc1
        doc_id = "doc" + fname.replace("doc-", "").replace(".txt", "")
        
        # 读取文档内容
        with open(os.path.join(corpus_dir, fname), "r", encoding="utf-8") as f:
            docs[doc_id] = f.read()
    
    return docs


def build_corpus_chunks(corpus_dir: str, chunk_size: int = 200, overlap: int = 40) -> List[Dict]:
    """将语料库目录中的所有文档切分成一个扁平的块池
    
    这个块池是后续检索操作的目标集合
    
    参数:
        corpus_dir: 语料库目录路径
        chunk_size: 每个块的字符数，默认 200
        overlap: 相邻块之间的重叠字符数，默认 40
    
    返回:
        所有文档的块列表，每个块包含 doc_id, start, end, text, chunk_id
    """
    # 加载所有文档
    docs = load_corpus(corpus_dir)
    all_chunks = []
    
    # 对每个文档进行分块
    for doc_id, text in docs.items():
        all_chunks.extend(fixed_length_chunks(text, doc_id, chunk_size, overlap))
    
    # 为每个块分配全局唯一的 chunk_id
    for i, c in enumerate(all_chunks):
        c["chunk_id"] = i
    
    return all_chunks
