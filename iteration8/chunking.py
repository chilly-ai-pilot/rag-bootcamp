"""
Iteration 2 分块策略模块

支持多种切块策略的对比实验：
1. fixed_200_40: 固定长度 200 字符，40 字符重叠（Iteration 1 baseline）
2. fixed_300_30: 固定长度 300 字符，30 字符重叠（更大块，减少信息割裂）
3. semantic: 按段落边界切分，保持语义完整性
4. fixed_100_50: 更小粒度 100 字符，50 字符重叠

所有策略保持统一的返回格式: [{doc_id, start, end, text, chunk_id}, ...]
"""
import os
import re
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


def semantic_chunks(text: str, doc_id: str, max_size: int = 120, overlap_size: int = 60) -> List[Dict]:
    """按语义边界（句子）分块，保持句子完整性 + 字符级重叠
    
    策略：
    - 每个块最多 100 字符（在句子边界处断块，不硬切断句子）
    - 相邻块之间 70 字符重叠（70%重叠率）
    - 使用多种分隔符切分句子，保持句子完整性
    - 提高检索覆盖度
    
    参数:
        text: 待分块的原始文本
        doc_id: 文档标识符
        max_size: 最大块大小（字符数），默认 100
        overlap_size: 重叠字符数，默认 70
    
    返回:
        字典列表: [{doc_id, start, end, text}, ...]
    """
    if not text:
        return []
    
    chunks = []
    
    # 按分隔符切分成句子单元（按优先级）
    separators = ["。", "！", "？", "；", "，", "\n", " "]
    
    def split_by_separator(txt: str, sep: str) -> list:
        """按分隔符切分，保留分隔符"""
        if sep not in txt:
            return [txt]
        parts = txt.split(sep)
        result = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                # 不是最后一个，加回分隔符
                result.append(part + sep)
            elif part:  # 最后一个且非空
                result.append(part)
        return result
    
    # 先按句号等主要标点切分
    sentences = [text]
    for sep in separators:
        new_sentences = []
        for sent in sentences:
            new_sentences.extend(split_by_separator(sent, sep))
        sentences = new_sentences
    
    # 去除空字符串
    sentences = [s for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    # 拼接块：累积句子直到接近 max_size
    buffer = ""
    buffer_start = 0
    current_pos = 0
    
    for sentence in sentences:
        # 找到句子在原文中的位置
        sentence_start = text.find(sentence, current_pos)
        if sentence_start == -1:
            # 找不到，可能是空格等，跳过
            continue
        
        # 初始化缓冲区
        if not buffer:
            buffer = sentence
            buffer_start = sentence_start
            current_pos = sentence_start + len(sentence)
            continue
        
        # 尝试加入当前句子
        if len(buffer) + len(sentence) <= max_size:
            # 可以加入
            buffer += sentence
            current_pos = sentence_start + len(sentence)
        else:
            # 超过上限，保存当前块
            chunk_end = buffer_start + len(buffer)
            chunks.append({
                "doc_id": doc_id,
                "start": buffer_start,
                "end": chunk_end,
                "text": buffer
            })
            
            # 计算重叠：从块尾部截取 overlap_size 字符
            if len(buffer) >= overlap_size:
                overlap_text = buffer[-overlap_size:]
                overlap_start = chunk_end - overlap_size
            else:
                # 块太小，全部作为重叠
                overlap_text = buffer
                overlap_start = buffer_start
            
            # 新缓冲区：重叠部分 + 当前句子
            buffer = overlap_text + sentence
            buffer_start = overlap_start
            current_pos = sentence_start + len(sentence)
    
    # 保存最后的缓冲区
    if buffer:
        chunk_end = buffer_start + len(buffer)
        chunks.append({
            "doc_id": doc_id,
            "start": buffer_start,
            "end": chunk_end,
            "text": buffer
        })
    
    return chunks


def small_overlap_chunks(text: str, doc_id: str, chunk_size: int = 100, overlap: int = 50) -> List[Dict]:
    """更小粒度的固定长度分块，更大的重叠比例
    
    相比 fixed_200_40，这个策略：
    - 块更小（100 vs 200），更细粒度
    - 重叠更多（50% vs 20%），减少边界切断问题
    
    参数:
        text: 待分块的原始文本
        doc_id: 文档标识符
        chunk_size: 每个块的字符数，默认 100
        overlap: 相邻块之间的重叠字符数，默认 50
    
    返回:
        字典列表: [{doc_id, start, end, text}, ...]
    """
    return fixed_length_chunks(text, doc_id, chunk_size, overlap)


def build_corpus_chunks(corpus_dir: str, strategy: str = 'fixed_200_40') -> List[Dict]:
    """将语料库目录中的所有文档切分成一个扁平的块池
    
    支持多种切块策略：
    - fixed_200_40: 200字符，40字符重叠（Iteration 1 baseline）
    - fixed_300_30: 300字符，30字符重叠（更大块，减少信息割裂）
    - semantic: 按段落边界切分
    - fixed_100_50: 100字符，50字符重叠
    
    参数:
        corpus_dir: 语料库目录路径
        strategy: 切块策略，可选值: 'fixed_200_40', 'fixed_300_30', 'semantic', 'fixed_100_50'
    
    返回:
        所有文档的块列表，每个块包含 doc_id, start, end, text, chunk_id
    """
    # 加载所有文档
    docs = load_corpus(corpus_dir)
    all_chunks = []
    
    # 根据策略选择切块函数
    if strategy == 'fixed_200_40':
        chunk_func = lambda text, doc_id: fixed_length_chunks(text, doc_id, 200, 40)
    elif strategy == 'fixed_300_30':
        chunk_func = lambda text, doc_id: fixed_length_chunks(text, doc_id, 300, 30)
    elif strategy == 'semantic':
        chunk_func = semantic_chunks
    elif strategy == 'fixed_100_50':
        chunk_func = lambda text, doc_id: small_overlap_chunks(text, doc_id, 100, 50)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    # 对每个文档进行分块
    for doc_id, text in docs.items():
        all_chunks.extend(chunk_func(text, doc_id))
    
    # 为每个块分配全局唯一的 chunk_id
    for i, c in enumerate(all_chunks):
        c["chunk_id"] = i
    
    return all_chunks
