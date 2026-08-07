"""
生成模块：使用 DeepSeek API 根据检索到的上下文片段生成答案

如果 DEEPSEEK_API_KEY 环境变量已设置且 openai 包已安装，则使用 DeepSeek API；
否则回退到模拟模式，这样整个流程（检索 -> 生成 -> 评分）仍可端到端运行，
无需网络访问或 API 密钥。

这个模拟回退是真实的设计选择，不只是开发便利性考虑：
Iteration 0 的验收标准是"流程端到端运行"，而不是"生成质量好"——
因此它不应该被缺失的凭证所阻塞。

Iteration 6: 添加拒答机制，当检索质量或生成质量不足时拒绝回答。
"""
import os
import asyncio
from typing import List, Dict, Optional

# 尝试导入 OpenAI SDK（DeepSeek 兼容 OpenAI API 格式）
try:
    from openai import OpenAI, AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ============ Iteration 6: 拒答机制 ============

def should_reject_by_judge_score(
    faithfulness: Optional[float],
    relevance: Optional[float],
    f_threshold: float = 0.80,
    r_threshold: float = 0.75
) -> bool:
    """Layer 3: 基于Judge评分判断是否拒答（保守模式）
    
    拒答条件（满足任一即拒答）：
    - Faithfulness < 0.80 (忠实度差，可能编造)
    - Relevance < 0.75 (相关性差，答非所问)
    
    参数:
        faithfulness: 忠实度分数 (0-1)
        relevance: 相关性分数 (0-1)
        f_threshold: 忠实度阈值，默认0.80
        r_threshold: 相关性阈值，默认0.75
    
    返回:
        True: 应该拒答
        False: 可以回答
    """
    if faithfulness is None or relevance is None:
        # 没有Judge分数，不拒答（让用户看到结果）
        return False
    
    # 保守模式：Faithfulness < 0.80 OR Relevance < 0.75
    return faithfulness < f_threshold or relevance < r_threshold


def should_reject_by_citation_coverage(
    raw_citations_count: int,
    valid_citations_count: int,
    coverage_threshold: float = 0.70
) -> bool:
    """Layer 2: 基于引用覆盖率判断是否拒答（检测引用幻觉）
    
    引用覆盖率 = 验证通过的citations / LLM声称的citations
    低覆盖率说明LLM产生了较多引用幻觉，应该拒答
    
    例如：
    - LLM生成3个引用，验证后只有2个有效 → 覆盖率 = 2/3 = 67%
    - LLM生成1个引用，验证后1个有效 → 覆盖率 = 1/1 = 100%
    
    拒答条件：
    - coverage < coverage_threshold (引用幻觉率过高)
    
    参数:
        raw_citations_count: LLM声称的引用数量（Step 2提取的citations数量）
        valid_citations_count: 验证通过的引用数量
        coverage_threshold: 覆盖率阈值，默认0.70（低于70%拒答）
    
    返回:
        True: 应该拒答
        False: 可以回答
    """
    if raw_citations_count == 0:
        # 没有引用，不通过覆盖率拒答（可能由其他层拒答）
        return False
    
    coverage = valid_citations_count / raw_citations_count
    return coverage < coverage_threshold


def should_reject_by_rerank_score(
    rerank_scores: Optional[List[float]],
    max_score_threshold: float = 0.75,
    top_n: int = 2,
    top_n_avg_threshold: float = 0.40
) -> bool:
    """Layer 1: 基于Rerank分数判断是否拒答（检索质量）
    
    拒答条件（满足任一即拒答）：
    - max(rerank_scores) < max_score_threshold (所有chunk中最高分都很低)
    - mean(rerank_scores[:top_n]) < top_n_avg_threshold (top-N的平均分很低)
    
    为什么改用max和topN：
    - top1分数不稳定（可能第1个不相关，但第2/3个很相关）
    - max分数表示"最好的chunk有多好"
    - topN平均表示"最好的N个chunk整体质量"
    - N可调：文档少时用1-2，文档多时用3-5
    
    参数:
        rerank_scores: rerank分数列表（降序排列）
        max_score_threshold: 最高分阈值，默认0.75
        top_n: topN的N值，默认2
        top_n_avg_threshold: topN平均分阈值，默认0.40
    
    返回:
        True: 应该拒答
        False: 可以回答
    """
    if not rerank_scores:
        # 没有rerank分数（如vector模式），不拒答
        return False
    
    # 检查max分数
    max_score = max(rerank_scores)
    if max_score < max_score_threshold:
        return True
    
    # 检查topN平均分数
    if len(rerank_scores) >= top_n:
        top_n_avg = sum(rerank_scores[:top_n]) / top_n
        if top_n_avg < top_n_avg_threshold:
            return True
    
    return False


def generate_rejection_answer() -> str:
    """生成拒答模板
    
    返回统一的拒答回复，避免编造内容。
    """
    return "抱歉，我在提供的资料中未找到足够充分的信息来准确回答您的问题。建议您查阅完整的产品手册或联系客服获取更详细的解答。"


# 系统提示词：限定模型只能基于给定资料作答
SYSTEM_PROMPT = (
    "你是一个只能根据给定资料回答问题的助手。"
    "只使用下面提供的资料片段作答，禁止使用资料之外的知识或猜测。"
    "如果资料片段中没有足够信息回答问题，必须回答\"未找到充分依据\"，不要编造。"
    "回答时用[文档X:片段N]标注引用来源，放在引用内容之后。每个片段只标注一次。"
)


def _build_context(retrieved_chunks):
    """将检索到的块列表构建为带编号的上下文字符串
    
    参数:
        retrieved_chunks: 检索到的文档块列表
    
    返回:
        格式化的上下文字符串，每个片段有编号标记
    """
    lines = []
    for i, c in enumerate(retrieved_chunks):
        doc_num = c["doc_id"].replace("doc", "")
        lines.append(f"[文档{doc_num}:片段{i+1}] {c['text']}")
    return "\n\n".join(lines)


def get_generator_client():
    """
    获取同步的 DeepSeek 客户端（用于向后兼容）
    
    返回:
        OpenAI 客户端实例或 None（如果不可用）
    """
    if not _OPENAI_AVAILABLE:
        return None
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )


def generate_answer(query: str, retrieved_chunks: list, model: str = "deepseek-chat", client=None) -> str:
    """根据检索到的上下文片段生成答案（同步版本）
    
    参数:
        query: 用户查询问题
        retrieved_chunks: 检索到的相关文档块列表
        model: 使用的模型名称，默认 "deepseek-chat"
        client: OpenAI 客户端（如果为 None，会自动创建）
    
    返回:
        生成的答案文本
    """
    # 如果没有检索到任何片段，直接返回
    if not retrieved_chunks:
        return "未找到充分依据"

    # 如果没有提供客户端，尝试创建
    if client is None:
        client = get_generator_client()
    
    # 如果客户端可用，使用 DeepSeek API
    if client:
        # 构建包含资料片段的提示词
        context = _build_context(retrieved_chunks)
        prompt = f"资料：\n{context}\n\n问题：{query}"
        
        # 调用 DeepSeek API 生成答案
        resp = client.chat.completions.create(
            model=model,
            max_tokens=500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
        )
        
        return resp.choices[0].message.content

    # 回退到模拟生成
    return _mock_generate(query, retrieved_chunks)


async def generate_answer_async(query: str, retrieved_chunks: list, model: str = "deepseek-chat", client=None) -> str:
    """根据检索到的上下文片段生成答案（异步版本）
    
    参数:
        query: 用户查询问题
        retrieved_chunks: 检索到的相关文档块列表
        model: 使用的模型名称，默认 "deepseek-chat"
        client: AsyncOpenAI 客户端（如果为 None，会自动创建）
    
    返回:
        生成的答案文本
    """
    # 如果没有检索到任何片段，直接返回
    if not retrieved_chunks:
        return "未找到充分依据"
    
    # 如果没有提供客户端，尝试创建
    if client is None:
        if not _OPENAI_AVAILABLE:
            return _mock_generate(query, retrieved_chunks)
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return _mock_generate(query, retrieved_chunks)
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    
    # 构建包含资料片段的提示词
    context = _build_context(retrieved_chunks)
    prompt = f"资料：\n{context}\n\n问题：{query}"
    
    # 调用 DeepSeek API 生成答案
    resp = await client.chat.completions.create(
        model=model,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
    )
    
    return resp.choices[0].message.content


def _mock_generate(query: str, retrieved_chunks: list) -> str:
    """Iteration 0 占位函数 — 没有设置 DEEPSEEK_API_KEY 或缺少依赖包
    
    允许在不消耗 API 调用的情况下验证 检索->评分 的连接
    
    参数:
        query: 用户查询问题
        retrieved_chunks: 检索到的文档块列表
    
    返回:
        模拟的响应信息
    """
    return f"[MOCK] 收到 {len(retrieved_chunks)} 个检索片段，未接入真实生成模型。"


# ============================================================
# Iteration 6: 使用 DeepSeek Chat 解决 citation hallucination
# ============================================================

async def generate_answer_async(
    query: str,
    retrieved_chunks: list,
    chunking_strategy: str,  # 必须传递！必须与检索时使用的strategy一致
    client=None,
    model: str = "deepseek-chat",
    rejection_config: Optional[Dict] = None
):
    """
    使用 DeepSeek Chat 生成带引用的答案（Iteration 6 - inline annotation 方案）
    
    改进策略（两步生成 + inline 标注）：
    - Step 1: 生成答案时直接在引用内容后标注 [文档X:片段N]
    - Step 2: LLM 看到 Step 1 的标注答案，提取每个标注对应的 span
    - Step 3: 程序验证并修正标注
    
    Iteration 6 拒答机制：
    - Layer 1 (Rerank): 检查rerank分数质量，检索质量不足时拒答
    - Layer 2 (Citation): 检查引用覆盖率，引用幻觉过多时拒答
    - Layer 3 (Judge): 生成后评估Faithfulness/Relevance，质量不足时拒答
    
    参数:
        query: 用户查询
        retrieved_chunks: 检索到的文档块
        chunking_strategy: chunking策略名称（必须与检索时使用的一致，用于建立正确的片段序号映射）
        client: AsyncOpenAI 客户端
        model: 模型名称（默认 deepseek-chat）
        rejection_config: 拒答配置字典
    
    返回:
        {
            "answer": "带引用标注的最终答案",
            "raw_answer": "原始答案（带标注）",
            "citations": [{span, source}, ...],
            "reasoning": None,
            "faithfulness_score": float,
            "relevance_score": float,
            "rejected": bool,
            "rejection_reason": str
        }
    """
    import json
    import re
    
    if not retrieved_chunks:
        return {
            "answer": "未找到充分依据",
            "raw_answer": "未找到充分依据",
            "citations": [],
            "reasoning": None,
            "faithfulness_score": None,
            "relevance_score": None,
            "rejected": True,
            "rejection_reason": "No retrieved chunks"
        }
    
    # ============================================================
    # Layer 1: Rerank拒答检查（在生成之前）
    # ============================================================
    rejected = False
    rejection_reason = None
    
    if rejection_config and rejection_config.get('rejection_enabled', False):
        layer1_cfg = rejection_config.get('rejection_layers', {}).get('layer1_rerank', {})
        
        if layer1_cfg.get('enabled', False):
            # 提取rerank分数
            rerank_scores = [c.get('rerank_score') for c in retrieved_chunks if 'rerank_score' in c]
            
            # 如果有rerank分数，检查是否应该拒答
            if rerank_scores:
                max_score_threshold = layer1_cfg.get('max_score_threshold', 0.75)
                top_n = layer1_cfg.get('top_n', 2)
                top_n_avg_threshold = layer1_cfg.get('top_n_avg_threshold', 0.40)
                
                # 检查拒答条件
                if should_reject_by_rerank_score(
                    rerank_scores,
                    max_score_threshold=max_score_threshold,
                    top_n=top_n,
                    top_n_avg_threshold=top_n_avg_threshold
                ):
                    rejected = True
                    max_score = max(rerank_scores) if rerank_scores else 0
                    top_n_avg = sum(rerank_scores[:top_n]) / top_n if len(rerank_scores) >= top_n else 0
                    rejection_reason = f"Low rerank quality (max={max_score:.3f}, top{top_n}_avg={top_n_avg:.3f})"
                    
                    # 直接返回拒答
                    rejection_message = rejection_config.get('rejection_message', generate_rejection_answer())
                    return {
                        "answer": rejection_message,
                        "raw_answer": rejection_message,
                        "citations": [],
                        "reasoning": None,
                        "faithfulness_score": None,
                        "relevance_score": None,
                        "rejected": True,
                        "rejection_reason": rejection_reason
                    }
    
    # 创建客户端
    if client is None:
        if not _OPENAI_AVAILABLE:
            return {
                "answer": "[MOCK] Inline annotation generation",
                "raw_answer": "[MOCK]",
                "citations": [],
                "reasoning": None
            }
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return {
                "answer": "[ERROR] DEEPSEEK_API_KEY not set",
                "raw_answer": "",
                "citations": [],
                "reasoning": None
            }
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    
    # 构建上下文，同时建立 source -> chunk 的映射
    context_lines = []
    source_to_chunk = {}
    
    # 首先，加载该chunking strategy下的所有chunks，建立完整的文档内序号映射
    # 这样即使retrieved_chunks只包含部分chunk，序号也能对应到文档中的正确位置
    from chunking import build_corpus_chunks
    all_chunks = build_corpus_chunks("corpus", strategy=chunking_strategy)
    
    # 为每个文档的chunks建立序号（基于全部chunks，不是retrieved_chunks）
    doc_chunk_indices = {}
    for chunk in all_chunks:
        doc_id = chunk['doc_id']
        if doc_id not in doc_chunk_indices:
            doc_chunk_indices[doc_id] = []
        doc_chunk_indices[doc_id].append(chunk['chunk_id'])
    
    # 为每个文档的chunk_id排序，建立 chunk_id -> 文档内序号 的映射
    chunk_id_to_doc_seq = {}
    for doc_id, chunk_ids in doc_chunk_indices.items():
        chunk_ids_sorted = sorted(chunk_ids)
        for seq, chunk_id in enumerate(chunk_ids_sorted, 1):
            chunk_id_to_doc_seq[chunk_id] = seq
    
    for i, c in enumerate(retrieved_chunks):
        doc_num = c["doc_id"].replace("doc", "")
        chunk_id = c.get('chunk_id', i)
        
        # 获取该chunk在其文档中的序号
        doc_seq = chunk_id_to_doc_seq.get(chunk_id, i + 1)
        
        label = f"文档{doc_num}:片段{doc_seq}"
        context_lines.append(f"[{label}] {c['text']}")
        
        # 保存 chunk 信息
        source_to_chunk[label] = {
            'text': c['text'],
            'doc_id': c['doc_id'],
            'doc_seq': doc_seq,  # 在该文档中的序号
            'chunk_id': chunk_id,  # 全局 chunk ID
            'full_id': f"{c['doc_id']}:chunk{doc_seq}"  # 如 "doc14:chunk5"（文档内序号）
        }
    
    context = "\n\n".join(context_lines)
    
    # ============================================================
    # Step 1: 生成答案，直接用 inline 标注引用来源
    # ============================================================
    step1_prompt = f"""资料：
{context}

问题：{query}

任务：根据资料回答问题，并在引用处标注来源。

【关键要求】：
1. 引用部分：必须从资料中【逐字逐句复制粘贴】，不改任何文字、标点、空格
   - 像 Ctrl+C / Ctrl+V 一样精确复制
   - 不要改写、不要合并句子、不要调整标点
   
2. 标注格式：引用内容后面立即加上 [文档X:片段N]
   例如："SmartLock-100 支持 35-60mm 门厚[文档1:片段5]。"

3. 解释部分：可以用自己的话补充说明，不需要标注

4. 如果资料不足，回答"未找到充分依据"

示例：
问：门锁支持多厚的门？
好的回答：SmartLock-100 支持 35-60mm 门厚[文档1:片段5]。需注意导向片宽度需 24mm。
（第一句摘抄+标注，第二句是补充说明）

只输出答案文本（带标注）。"""
    
    try:
        # 调用 Step 1
        resp1 = await client.chat.completions.create(
            model=model,
            max_tokens=500,
            messages=[
                {"role": "system", "content": "你是一个只根据给定资料回答问题的助手。"},
                {"role": "user", "content": step1_prompt}
            ]
        )
        
        raw_answer_with_tags = resp1.choices[0].message.content.strip()
        
        # 如果答案太短或表示未找到，直接返回
        if not raw_answer_with_tags or raw_answer_with_tags == "未找到充分依据" or len(raw_answer_with_tags) < 5:
            return {
                "answer": raw_answer_with_tags,
                "raw_answer": raw_answer_with_tags,
                "citations": [],
                "reasoning": None
            }
        
    except Exception as e:
        print(f"❌ Step 1 (answer generation) failed: {e}")
        return {
            "answer": f"[ERROR Step 1] {e}",
            "raw_answer": "",
            "citations": [],
            "reasoning": None
        }
    
    # ============================================================
    # Step 2: LLM 提取每个标注对应的 span
    # ============================================================
    step2_prompt = f"""我生成了一个带引用标注的答案，现在需要你提取每个标注对应的引用内容。

资料片段：
{context}

带标注的答案：
{raw_answer_with_tags}

任务：
找出答案中所有的 [文档X:片段N] 标注，并告诉我每个标注对应的是哪段引用内容。

输出格式（JSON）：
{{
  "citations": [
    {{"span": "被标注的引用内容（逐字复制）", "source": "文档X:片段N"}}
  ]
}}

【严格规则】：
- span 是答案中被 [文档X:片段N] 标注的那段文字
- **span 中不要包含 [文档X:片段N] 标注符号本身**
- span 必须逐字复制，不要改动
- 按照标注在答案中出现的顺序列出
- 如果没有标注，citations 为空数组 []

【示例】：
答案："SmartLock-100 支持 35-60mm 门厚[文档1:片段5]。需注意导向片宽度。"
正确的输出：
{{
  "citations": [
    {{"span": "SmartLock-100 支持 35-60mm 门厚", "source": "文档1:片段5"}}
  ]
}}

错误的输出（不要这样做）：
{{
  "citations": [
    {{"span": "SmartLock-100 支持 35-60mm 门厚[文档1:片段5]", "source": "文档1:片段5"}}
  ]
}}

只输出 JSON。"""
    
    try:
        # 调用 Step 2
        resp2 = await client.chat.completions.create(
            model=model,
            max_tokens=800,
            response_format={"type": "json_object"},  # 强制 JSON 输出
            messages=[
                {"role": "user", "content": step2_prompt}
            ]
        )
        
        raw_response = resp2.choices[0].message.content
        
        # 解析 JSON
        result = json.loads(raw_response)
        citations = result.get("citations", [])
        raw_citations_count = len(citations)  # LLM声称的引用数量
        
        # 为每个 citation 添加实际的 chunk text（用于后续验证）
        valid_citations = []
        for cit in citations:
            source = cit.get("source", "")
            span = cit.get("span", "")
            
            # 直接从映射中查找
            if source in source_to_chunk:
                chunk_info = source_to_chunk[source]
                cit['chunk_text'] = chunk_info['text']
                cit['chunk_full_id'] = chunk_info['full_id']  # 保存 "doc3:chunk5" 格式
                
                # 验证：span是否真的在chunk_text中
                chunk_text = chunk_info['text']
                
                # 验证逻辑：
                # 1. 精确匹配：span完全在chunk中（最可靠）
                # 2. 子串匹配：去掉标点空格后匹配（容忍格式差异）
                # 3. 字符级别重叠：使用字符集重叠（适用于中文）
                # 4. Embedding相似度：语义相似度检查（捕获总结性引用）
                
                is_valid = False
                validation_method = None
                
                # 方法1：精确匹配
                if span in chunk_text:
                    is_valid = True
                    validation_method = "exact_match"
                
                # 方法2：容忍标点和空格差异
                elif span and chunk_text:
                    import re
                    # 移除所有标点和空格
                    span_clean = re.sub(r'[^\w]', '', span)
                    chunk_clean = re.sub(r'[^\w]', '', chunk_text)
                    if span_clean and span_clean in chunk_clean:
                        is_valid = True
                        validation_method = "punctuation_tolerant"
                
                # 方法3：字符级别重叠（适用于中文，要求50%重叠）
                if not is_valid and len(span) > 10:
                    # 对于较长的span，检查字符重叠率
                    span_chars = set(span)
                    chunk_chars = set(chunk_text)
                    overlap_ratio = len(span_chars & chunk_chars) / len(span_chars)
                    if overlap_ratio > 0.5:  # 降低到50%，容忍更多改写
                        is_valid = True
                        validation_method = "character_overlap"
                
                # 方法4：Embedding相似度（捕获总结性引用，只在前3种方法都失败时使用）
                if not is_valid and len(span) > 15:  # 降低到15，覆盖更多短span的总结性引用
                    try:
                        from retrieval import _get_embedding_model
                        embedding_model = _get_embedding_model()
                        
                        # 计算span和chunk的embedding相似度
                        span_embedding = embedding_model.encode([span])[0]
                        chunk_embedding = embedding_model.encode([chunk_text])[0]
                        
                        # 计算余弦相似度
                        from numpy import dot
                        from numpy.linalg import norm
                        similarity = dot(span_embedding, chunk_embedding) / (norm(span_embedding) * norm(chunk_embedding))
                        
                        # 阈值设为0.50（短span改写的相似度通常在0.5-0.7之间）
                        if similarity > 0.50:
                            is_valid = True
                            validation_method = f"embedding_similarity({similarity:.3f})"
                            print(f"✅ Citation validated by embedding similarity: {similarity:.3f} (source={source})")
                    except Exception as e:
                        # Embedding验证失败不影响结果，继续使用is_valid=False
                        pass  # 静默忽略embedding验证失败
                
                if is_valid:
                    cit['validation_method'] = validation_method  # 记录验证方法
                    valid_citations.append(cit)
                else:
                    # span不在chunk中，可能是幻觉
                    print(f"⚠️  Citation validation failed: span not in chunk (source={source})")
                    print(f"   Span: {span[:100]}...")
                    print(f"   Chunk: {chunk_text[:100]}...")
            else:
                # source不存在，肯定是幻觉
                print(f"⚠️  Citation validation failed: source not found ({source})")
        
        # 更新citations为验证通过的
        citations = valid_citations
        valid_citations_count = len(citations)
        
    except Exception as e:
        print(f"❌ Step 2 (citation extraction) failed: {e}")
        print(f"Raw response: {raw_response if 'raw_response' in locals() else 'N/A'}")
        # Step 2 失败不影响答案，只是没有引用标注
        # 移除所有标注，返回纯文本
        clean_answer = re.sub(r'\[文档\d+:片段\d+\]', '', raw_answer_with_tags)
        return {
            "answer": clean_answer,
            "raw_answer": clean_answer,
            "citations": [],
            "reasoning": None
        }
    
    # ============================================================
    # Layer 2: 引用覆盖率检查（检测引用幻觉）
    # ============================================================
    # 在这里我们已经知道了 raw_citations_count 和 valid_citations_count
    # 但需要从rejection_config读取配置
    
    if rejection_config and rejection_config.get('rejection_enabled', False):
        layer2_cfg = rejection_config.get('rejection_layers', {}).get('layer2_citation', {})
        
        if layer2_cfg.get('enabled', False):
            coverage_threshold = layer2_cfg.get('coverage_threshold', 0.70)
            
            if should_reject_by_citation_coverage(
                raw_citations_count,
                valid_citations_count,
                coverage_threshold=coverage_threshold
            ):
                rejected = True
                coverage = valid_citations_count / raw_citations_count if raw_citations_count > 0 else 0
                rejection_reason = f"Low citation coverage ({valid_citations_count}/{raw_citations_count} = {coverage:.1%})"
                
                # 直接返回拒答
                rejection_message = rejection_config.get('rejection_message', generate_rejection_answer())
                return {
                    "answer": rejection_message,
                    "raw_answer": rejection_message,
                    "citations": [],
                    "reasoning": None,
                    "faithfulness_score": None,
                    "relevance_score": None,
                    "rejected": True,
                    "rejection_reason": rejection_reason
                }
    
    # ============================================================
    # Iteration 6: 内置Judge评估和拒答机制
    # ============================================================
    raw_answer_clean = re.sub(r'\[文档\d+:片段\d+\]', '', raw_answer_with_tags)
    
    # 如果启用拒答机制，调用Judge评估
    faithfulness_score = None
    relevance_score = None
    rejected = False
    rejection_reason = None
    
    # 从外部传入的rejection_config（如果有）
    rejection_config = locals().get('rejection_config')
    
    if rejection_config and rejection_config.get('rejection_enabled', False):
        try:
            # 导入Judge评估函数
            from evaluation import llm_combined_check_async
            
            # 获取Judge配置
            judge_model = rejection_config.get('judge_model', 'deepseek-chat')
            judge_base_url = rejection_config.get('judge_base_url', 'https://api.deepseek.com')
            judge_api_key = rejection_config.get('judge_api_key') or os.getenv('DEEPSEEK_API_KEY')
            
            if judge_api_key:
                # 创建Judge客户端（与generator分离）
                judge_client = AsyncOpenAI(
                    api_key=judge_api_key,
                    base_url=judge_base_url
                )
                
                # 调用Judge评估
                judge_result = await llm_combined_check_async(
                    query,
                    raw_answer_clean,
                    retrieved_chunks,
                    judge_client,
                    judge_model
                )
                
                await judge_client.close()
                
                # 提取分数
                import re as re_module
                response_text = judge_result['raw_response']
                
                # 提取Faithfulness分数
                faith_match = re_module.search(r'【Faithfulness 分数】\s*\n?\s*([0-9.]+)', response_text)
                if faith_match:
                    faithfulness_score = float(faith_match.group(1))
                    if faithfulness_score > 1:
                        faithfulness_score /= 100
                
                # 提取Relevance分数
                rel_match = re_module.search(r'【Relevance 分数】\s*\n?\s*([0-9.]+)', response_text)
                if rel_match:
                    relevance_score = float(rel_match.group(1))
                    if relevance_score > 1:
                        relevance_score /= 100
                
                # 检查是否需要拒答
                layer3_cfg = rejection_config['rejection_layers']['layer3_judge']
                if layer3_cfg['enabled']:
                    f_threshold = layer3_cfg['faithfulness_threshold']
                    r_threshold = layer3_cfg['relevance_threshold']
                    
                    if (faithfulness_score is not None and faithfulness_score < f_threshold) or \
                       (relevance_score is not None and relevance_score < r_threshold):
                        rejected = True
                        rejection_reason = f"Low quality (F={faithfulness_score:.2f}, R={relevance_score:.2f})"
                        
                        # 替换答案为拒答消息
                        rejection_message = rejection_config.get('rejection_message', generate_rejection_answer())
                        raw_answer_with_tags = rejection_message
                        raw_answer_clean = rejection_message
                        citations = []  # 拒答时清空引用
        
        except Exception as e:
            print(f"⚠️  Judge evaluation in generator failed: {e}")
            # Judge失败不影响生成，继续返回原答案
    
    # ============================================================
    # 返回结果
    # ============================================================
    return {
        "answer": raw_answer_with_tags,  # 保留原始标注（或拒答消息）
        "raw_answer": raw_answer_clean,   # 去掉标注的纯文本（或拒答消息）
        "citations": citations,
        "reasoning": None,  # chat 模型没有 reasoning
        "faithfulness_score": faithfulness_score,
        "relevance_score": relevance_score,
        "rejected": rejected,
        "rejection_reason": rejection_reason
    }
