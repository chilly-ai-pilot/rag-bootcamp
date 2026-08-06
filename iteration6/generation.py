"""
生成模块：使用 DeepSeek API 根据检索到的上下文片段生成答案

如果 DEEPSEEK_API_KEY 环境变量已设置且 openai 包已安装，则使用 DeepSeek API；
否则回退到模拟模式，这样整个流程（检索 -> 生成 -> 评分）仍可端到端运行，
无需网络访问或 API 密钥。

这个模拟回退是真实的设计选择，不只是开发便利性考虑：
Iteration 0 的验收标准是"流程端到端运行"，而不是"生成质量好"——
因此它不应该被缺失的凭证所阻塞。
"""
import os
import asyncio

# 尝试导入 OpenAI SDK（DeepSeek 兼容 OpenAI API 格式）
try:
    from openai import OpenAI, AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

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
# Iteration 6: 使用 DeepSeek Reasoner 解决 citation hallucination
# ============================================================

async def generate_answer_v6_async(
    query: str,
    retrieved_chunks: list,
    client=None,
    model: str = "deepseek-chat"  # 改回 chat 模型，更稳定
):
    """
    使用 DeepSeek Chat 生成带引用的答案（Iteration 6 - inline annotation 方案）
    
    改进策略（两步生成 + inline 标注）：
    - Step 1: 生成答案时直接在引用内容后标注 [文档X:片段N]
    - Step 2: LLM 看到 Step 1 的标注答案，提取每个标注对应的 span
    - Step 3: 程序验证并修正标注
    
    这样避免了"合并不相邻句子"的问题，每个引用位置独立标注
    
    参数:
        query: 用户查询
        retrieved_chunks: 检索到的文档块
        client: AsyncOpenAI 客户端
        model: 模型名称（默认 deepseek-chat）
    
    返回:
        {
            "answer": "带引用标注的最终答案",
            "raw_answer": "原始答案（带标注）",
            "citations": [{span, source}, ...],
            "reasoning": None
        }
    """
    import json
    import re
    
    if not retrieved_chunks:
        return {
            "answer": "未找到充分依据",
            "raw_answer": "未找到充分依据",
            "citations": [],
            "reasoning": None
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
    
    # 首先，为每个文档的chunks建立序号映射
    # 需要知道每个chunk在其文档中的相对位置
    from chunking import build_corpus_chunks
    all_chunks = build_corpus_chunks("corpus", strategy="fixed_100_50")  # 获取所有chunks以建立映射
    
    # 为每个文档的chunks建立序号
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
        
        # 为每个 citation 添加实际的 chunk text（用于后续验证）
        for cit in citations:
            source = cit.get("source", "")
            # 直接从映射中查找
            if source in source_to_chunk:
                chunk_info = source_to_chunk[source]
                cit['chunk_text'] = chunk_info['text']
                cit['chunk_full_id'] = chunk_info['full_id']  # 保存 "doc3:chunk5" 格式
        
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
    # 返回结果（验证和修正会在 run_eval.py 中统一处理）
    # ============================================================
    # 这里的 answer 保持 Step 1 的标注格式
    # raw_answer 去掉标注（用于验证）
    raw_answer_clean = re.sub(r'\[文档\d+:片段\d+\]', '', raw_answer_with_tags)
    
    return {
        "answer": raw_answer_with_tags,  # 保留原始标注
        "raw_answer": raw_answer_clean,   # 去掉标注的纯文本
        "citations": citations,
        "reasoning": None  # chat 模型没有 reasoning
    }
