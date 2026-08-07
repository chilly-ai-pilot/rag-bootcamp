"""
Iteration 5: LLM-as-Judge 评估模块

使用 LLM 评估 RAG 系统的质量指标：
1. Faithfulness（忠实度）：答案是否只基于检索到的文档，需要 {检索文档 + 生成答案}
2. Answer Relevance（相关性）：答案是否真正回答了用户问题，需要 {用户问题 + 生成答案}

每个指标都提供两种实现：
- llm_xxx_check: 直接使用 LLM 进行评估（推荐，更灵活）
- ragas_xxx_check: 使用 Ragas 框架（对比实验用）

注意：Ragas 相关的导入被延迟到函数内部，避免在不需要时加载重依赖
"""

import os
import asyncio
from typing import List, Dict
from openai import OpenAI


def _build_context(retrieved_chunks):
    """
    使用和 generation.py 完全一样的格式构建上下文
    
    这样 Judge 看到的文档格式和生成器一致
    """
    lines = []
    for i, c in enumerate(retrieved_chunks):
        doc_num = c.get("doc_id", "").replace("doc", "") or str(i+1)
        lines.append(f"[文档{doc_num}:片段{i+1}] {c['text']}")
    return "\n\n".join(lines)


def get_judge_llm():
    """
    获取 Judge LLM（使用阿里云 Qwen 模型）
    
    注意：返回 OpenAI 客户端，不依赖 LangChain
    """
    api_key = os.getenv("ALI_API_KEY")
    base_url = os.getenv("ALI_BASE_URL")
    
    if not api_key or not base_url:
        raise ValueError("请设置环境变量 ALI_API_KEY 和 ALI_BASE_URL")
    
    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )


def get_judge_embeddings():
    """
    获取 Judge 使用的 Embeddings（用于 answer_relevancy 指标）
    
    注意：需要 LangChain，延迟导入
    """
    from langchain_openai import OpenAIEmbeddings
    
    return OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=os.getenv("ALI_API_KEY", "dummy"),
        base_url=os.getenv("ALI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )


def evaluate_single_response(
    query: str,
    retrieved_contexts: List[str],
    generated_answer: str,
    ground_truth: str = None
) -> Dict[str, float]:
    """
    评估单条 RAG 响应（使用 Ragas）
    
    注意：需要 Ragas 依赖，延迟导入
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from langchain_openai import ChatOpenAI
    
    # 构造 Ragas 需要的数据格式
    data = {
        "question": [query],
        "answer": [generated_answer],
        "contexts": [retrieved_contexts],
    }
    
    if ground_truth:
        data["ground_truth"] = [ground_truth]
    
    dataset = Dataset.from_dict(data)
    
    # 配置 Judge LLM
    api_key = os.getenv("ALI_API_KEY")
    base_url = os.getenv("ALI_BASE_URL")
    
    judge_llm = ChatOpenAI(
        model="qwen-plus",
        api_key=api_key,
        base_url=base_url,
        temperature=0.0
    )
    
    judge_embeddings = get_judge_embeddings()
    
    # 评估指标
    metrics = [faithfulness, answer_relevancy]
    
    # 执行评估
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings
    )
    
    return {
        'faithfulness': result['faithfulness'],
        'answer_relevancy': result['answer_relevancy']
    }


def evaluate_batch(
    queries: List[str],
    contexts_list: List[List[str]],
    answers: List[str],
    ground_truths: List[str] = None
) -> Dict[str, List[float]]:
    """
    批量评估多条 RAG 响应（使用 Ragas）
    
    注意：需要 Ragas 依赖，延迟导入
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from langchain_openai import ChatOpenAI
    
    data = {
        "question": queries,
        "answer": answers,
        "contexts": contexts_list,
    }
    
    if ground_truths:
        data["ground_truth"] = ground_truths
    
    dataset = Dataset.from_dict(data)
    
    api_key = os.getenv("ALI_API_KEY")
    base_url = os.getenv("ALI_BASE_URL")
    
    judge_llm = ChatOpenAI(
        model="qwen-plus",
        api_key=api_key,
        base_url=base_url,
        temperature=0.0
    )
    
    judge_embeddings = get_judge_embeddings()
    
    metrics = [faithfulness, answer_relevancy]
    
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings
    )
    
    return {
        'faithfulness': result['faithfulness'],
        'answer_relevancy': result['answer_relevancy']
    }


def llm_faithfulness_check(
    generated_answer: str,
    retrieved_chunks: List[Dict],
    llm=None
) -> Dict:
    """
    简化版的 Faithfulness 检查（不使用 Ragas，直接调用 LLM）
    
    用于快速测试或作为 Ragas 的备选方案
    
    参数:
        generated_answer: 生成的答案
        retrieved_chunks: 检索到的完整文档块列表（包含 doc_id 和 text）
        llm: OpenAI 客户端（如果为 None，会自动创建）
    
    返回:
        {
            'raw_response': str,  # Judge 的完整响应
            'prompt': str  # 使用的 prompt
        }
    """
    if llm is None:
        llm = get_judge_llm()
    
    # 使用和 generation.py 完全一样的格式构建上下文
    contexts_text = _build_context(retrieved_chunks)
    
    prompt = f"""你是一个严格的事实核查员。请判断以下"生成答案"中的每个陈述是否都能在"检索文档"中找到依据。

【检索文档】
{contexts_text}

【生成答案】
{generated_answer}

请逐句分析生成答案，判断：
1. 哪些陈述有文档依据（标注来源）
2. 哪些陈述没有依据（可能是编造、推测或混淆其他产品）

最后给出一个 0-1 的分数：
- 1.0 表示所有陈述都有依据
- 0.0 表示完全编造
- 中间值表示部分有依据

请用以下格式回答：
【逐句分析】
...

【Faithfulness 分数】
0.XX

【理由】
...
"""
    
    response = llm.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    return {
        'raw_response': response.choices[0].message.content,
        'prompt': prompt
    }


async def llm_faithfulness_check_async(
    generated_answer: str,
    retrieved_chunks: List[Dict],
    client=None,
    model: str = "qwen-plus"
) -> Dict:
    """
    异步版本的 Faithfulness 检查，用于并发评估
    
    参数:
        generated_answer: 生成的答案
        retrieved_chunks: 检索到的完整文档块列表
        client: AsyncOpenAI 客户端（如果为 None，会自动创建）
        model: 使用的模型名称（qwen-plus 或 deepseek-chat）
    
    返回:
        {
            'raw_response': str,
            'prompt': str
        }
    """
    from openai import AsyncOpenAI
    
    if client is None:
        api_key = os.getenv("ALI_API_KEY")
        base_url = os.getenv("ALI_BASE_URL")
        
        if not api_key or not base_url:
            raise ValueError("请设置环境变量 ALI_API_KEY 和 ALI_BASE_URL")
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    contexts_text = _build_context(retrieved_chunks)
    
    prompt = f"""你是一个严格的事实核查员。请判断以下"生成答案"中的每个陈述是否都能在"检索文档"中找到依据。

【检索文档】
{contexts_text}

【生成答案】
{generated_answer}

请逐句分析生成答案，判断：
1. 哪些陈述有文档依据（标注来源）
2. 哪些陈述没有依据（可能是编造、推测或混淆其他产品）

最后给出一个 0-1 的分数：
- 1.0 表示所有陈述都有依据
- 0.0 表示完全编造
- 中间值表示部分有依据

请用以下格式回答：
【逐句分析】
...

【Faithfulness 分数】
0.XX

【理由】
...
"""
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    return {
        'raw_response': response.choices[0].message.content,
        'prompt': prompt
    }


def ragas_faithfulness_check(
    query: str,
    generated_answer: str,
    retrieved_chunks: List[Dict]
) -> Dict:
    """
    使用 Ragas 框架评估 Faithfulness（同步版本）
    
    注意：这是对比实验用的标准Ragas实现
    
    参数:
        query: 用户查询
        generated_answer: 生成的答案
        retrieved_chunks: 检索到的完整文档块列表
    
    返回:
        {
            'faithfulness_score': float,
            'raw_result': dict  # Ragas原始结果
        }
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError as e:
        raise ImportError(f"Ragas依赖缺失: {e}. 请运行: pip install ragas langchain-community langchain-openai datasets")
    
    # 提取纯文本contexts（Ragas需要的格式）
    contexts = [chunk['text'] for chunk in retrieved_chunks]
    
    # 构造Ragas数据集格式
    data = {
        "question": [query],
        "answer": [generated_answer],
        "contexts": [contexts],
    }
    dataset = Dataset.from_dict(data)
    
    # 配置Judge LLM
    api_key = os.getenv("ALI_API_KEY")
    base_url = os.getenv("ALI_BASE_URL")
    
    if not api_key or not base_url:
        raise ValueError("请设置环境变量 ALI_API_KEY 和 ALI_BASE_URL")
    
    judge_llm = ChatOpenAI(
        model="qwen-plus",
        api_key=api_key,
        base_url=base_url,
        temperature=0.0
    )
    
    judge_embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=api_key,
        base_url=base_url
    )
    
    # 执行Ragas评估
    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness],
            llm=judge_llm,
            embeddings=judge_embeddings
        )
        
        # Ragas返回的是列表，取第一个值
        faith_score = result['faithfulness']
        if isinstance(faith_score, (list, tuple)):
            faith_score = faith_score[0] if len(faith_score) > 0 else 0.5
        
        return {
            'faithfulness_score': float(faith_score),
            'ragas_response': str(result),
            'raw_result': result
        }
    except Exception as e:
        # Ragas内部可能有各种问题，返回详细错误
        raise RuntimeError(f"Ragas evaluation failed: {str(e)}")


async def ragas_faithfulness_check_async(
    query: str,
    generated_answer: str,
    retrieved_chunks: List[Dict]
) -> Dict:
    """
    使用 Ragas 框架评估 Faithfulness（异步版本）
    
    通过 asyncio.to_thread 将同步的 Ragas 调用转换为异步
    
    参数:
        query: 用户查询
        generated_answer: 生成的答案
        retrieved_chunks: 检索到的完整文档块列表
    
    返回:
        {
            'faithfulness_score': float,
            'ragas_response': str,
            'raw_result': dict
        }
    """
    # 在线程池中运行同步的 ragas_faithfulness_check
    return await asyncio.to_thread(
        ragas_faithfulness_check,
        query,
        generated_answer,
        retrieved_chunks
    )


def llm_combined_check(
    query: str,
    generated_answer: str,
    retrieved_chunks: List[Dict],
    llm=None
) -> Dict:
    """
    组合评估：一次请求同时评估 Faithfulness 和 Answer Relevance
    
    节省 API 调用成本，一个请求返回两个指标
    
    参数:
        query: 用户的问题
        generated_answer: 生成的答案
        retrieved_chunks: 检索到的完整文档块列表
        llm: OpenAI 客户端（如果为 None，会自动创建）
    
    返回:
        {
            'raw_response': str,
            'prompt': str
        }
    """
    if llm is None:
        llm = get_judge_llm()
    
    contexts_text = _build_context(retrieved_chunks)
    
    prompt = f"""你是一个专业的 RAG 系统评估员。请对以下问答进行双重评估：

【用户问题】
{query}

【检索文档】
{contexts_text}

【生成答案】
{generated_answer}

请完成两项评估：

## 评估一：Faithfulness（忠实度）
判断生成答案中的每个陈述是否都能在检索文档中找到依据。

评估要点：
- 哪些陈述有文档依据（标注来源）
- 哪些陈述没有依据（可能是编造、推测或混淆其他产品）

评分标准：
- 1.0 表示所有陈述都有依据
- 0.0 表示完全编造
- 中间值表示部分有依据

## 评估二：Answer Relevance（相关性）
判断生成答案是否真正回答了用户问题。

评估要点：
- **直接性**: 是否直接回答问题核心？
- **完整性**: 是否涵盖所有关键信息？
- **准确性**: 是否精确匹配问题内容？
- **无关内容**: 是否包含冗余信息？

评分标准：
- 1.0 表示完全相关，准确回答
- 0.8-0.9 表示基本相关，略有瑕疵
- 0.5-0.7 表示部分相关
- 0.3-0.4 表示弱相关
- 0.0-0.2 表示不相关

请用以下格式回答：

【Faithfulness 分析】
...

【Faithfulness 分数】
0.XX

【Faithfulness 理由】
...

【Relevance 分析】
...

【Relevance 分数】
0.XX

【Relevance 理由】
...
"""
    
    response = llm.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    return {
        'raw_response': response.choices[0].message.content,
        'prompt': prompt
    }


async def llm_combined_check_async(
    query: str,
    generated_answer: str,
    retrieved_chunks: List[Dict],
    client=None,
    model: str = "qwen-plus"
) -> Dict:
    """
    异步版本的组合评估
    
    参数:
        query: 用户的问题
        generated_answer: 生成的答案
        retrieved_chunks: 检索到的完整文档块列表
        client: AsyncOpenAI 客户端
        model: 使用的模型名称（qwen-plus 或 deepseek-chat）
    
    返回:
        {
            'raw_response': str,
            'prompt': str
        }
    """
    from openai import AsyncOpenAI
    
    if client is None:
        api_key = os.getenv("ALI_API_KEY")
        base_url = os.getenv("ALI_BASE_URL")
        
        if not api_key or not base_url:
            raise ValueError("请设置环境变量 ALI_API_KEY 和 ALI_BASE_URL")
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    contexts_text = _build_context(retrieved_chunks)
    
    prompt = f"""你是一个专业的 RAG 系统评估员。请对以下问答进行双重评估：

【用户问题】
{query}

【检索文档】
{contexts_text}

【生成答案】
{generated_answer}

请完成两项评估：

## 评估一：Faithfulness（忠实度）
判断生成答案中的每个陈述是否都能在检索文档中找到依据。

评估要点：
- 哪些陈述有文档依据（标注来源）
- 哪些陈述没有依据（可能是编造、推测或混淆其他产品）

评分标准：
- 1.0 表示所有陈述都有依据
- 0.0 表示完全编造
- 中间值表示部分有依据

## 评估二：Answer Relevance（相关性）
判断生成答案是否真正回答了用户问题。

评估要点：
- **直接性**: 是否直接回答问题核心？
- **完整性**: 是否涵盖所有关键信息？
- **准确性**: 是否精确匹配问题内容？
- **无关内容**: 是否包含冗余信息？

评分标准：
- 1.0 表示完全相关，准确回答
- 0.8-0.9 表示基本相关，略有瑕疵
- 0.5-0.7 表示部分相关
- 0.3-0.4 表示弱相关
- 0.0-0.2 表示不相关

请用以下格式回答：

【Faithfulness 分析】
...

【Faithfulness 分数】
0.XX

【Faithfulness 理由】
...

【Relevance 分析】
...

【Relevance 分数】
0.XX

【Relevance 理由】
...
"""
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    return {
        'raw_response': response.choices[0].message.content,
        'prompt': prompt
    }


def llm_relevance_check(
    query: str,
    generated_answer: str,
    llm=None
) -> Dict:
    """
    Answer Relevance 检查（不使用 Ragas，直接调用 LLM）
    
    评估生成的答案是否真正回答了用户的问题
    
    注意：推荐使用 llm_combined_check 来节省 API 成本
    
    参数:
        query: 用户的问题
        generated_answer: 生成的答案
        llm: OpenAI 客户端（如果为 None，会自动创建）
    
    返回:
        {
            'raw_response': str,  # Judge 的完整响应
            'prompt': str  # 使用的 prompt
        }
    """
    if llm is None:
        llm = get_judge_llm()
    
    prompt = f"""你是一个专业的问答质量评估员。请判断以下"生成答案"是否真正回答了"用户问题"。

【用户问题】
{query}

【生成答案】
{generated_answer}

请评估以下几个方面：
1. **直接性**: 答案是否直接回答了问题的核心关切？
2. **完整性**: 答案是否涵盖了问题要求的所有关键信息？
3. **准确性**: 答案是否精确匹配问题的具体询问内容（型号、参数等）？
4. **无关内容**: 答案是否包含了与问题无关的冗余信息？

特殊情况：
- 如果答案是"未找到充分依据"类型的回应，需要判断这是否是对该问题的合理回答
- 如果问题询问具体信息（如"电池型号"），答案必须提供该具体信息才算相关

最后给出一个 0-1 的分数：
- 1.0 表示完全相关，准确回答了问题
- 0.8-0.9 表示基本相关，可能缺少部分细节或有少量冗余
- 0.5-0.7 表示部分相关，回答了问题但不够精确或包含较多无关内容
- 0.3-0.4 表示弱相关，答案方向正确但未真正解决问题
- 0.0-0.2 表示不相关，答案完全偏离问题

请用以下格式回答：
【相关性分析】
...

【Answer Relevance 分数】
0.XX

【理由】
...
"""
    
    response = llm.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    return {
        'raw_response': response.choices[0].message.content,
        'prompt': prompt
    }


async def llm_relevance_check_async(
    query: str,
    generated_answer: str,
    client=None,
    model: str = "qwen-plus"
) -> Dict:
    """
    异步版本的 Answer Relevance 检查，用于并发评估
    
    参数:
        query: 用户的问题
        generated_answer: 生成的答案
        client: AsyncOpenAI 客户端（如果为 None，会自动创建）
        model: 使用的模型名称（qwen-plus 或 deepseek-chat）
    
    返回:
        {
            'raw_response': str,
            'prompt': str
        }
    """
    from openai import AsyncOpenAI
    
    if client is None:
        api_key = os.getenv("ALI_API_KEY")
        base_url = os.getenv("ALI_BASE_URL")
        
        if not api_key or not base_url:
            raise ValueError("请设置环境变量 ALI_API_KEY 和 ALI_BASE_URL")
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    prompt = f"""你是一个专业的问答质量评估员。请判断以下"生成答案"是否真正回答了"用户问题"。

【用户问题】
{query}

【生成答案】
{generated_answer}

请评估以下几个方面：
1. **直接性**: 答案是否直接回答了问题的核心关切？
2. **完整性**: 答案是否涵盖了问题要求的所有关键信息？
3. **准确性**: 答案是否精确匹配问题的具体询问内容（型号、参数等）？
4. **无关内容**: 答案是否包含了与问题无关的冗余信息？

特殊情况：
- 如果答案是"未找到充分依据"类型的回应，需要判断这是否是对该问题的合理回答
- 如果问题询问具体信息（如"电池型号"），答案必须提供该具体信息才算相关

最后给出一个 0-1 的分数：
- 1.0 表示完全相关，准确回答了问题
- 0.8-0.9 表示基本相关，可能缺少部分细节或有少量冗余
- 0.5-0.7 表示部分相关，回答了问题但不够精确或包含较多无关内容
- 0.3-0.4 表示弱相关，答案方向正确但未真正解决问题
- 0.0-0.2 表示不相关，答案完全偏离问题

请用以下格式回答：
【相关性分析】
...

【Answer Relevance 分数】
0.XX

【理由】
...
"""
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    return {
        'raw_response': response.choices[0].message.content,
        'prompt': prompt
    }


# 测试代码
if __name__ == "__main__":
    print("🧪 测试 Judge LLM 连接...")
    
    try:
        llm = get_judge_llm()
        response = llm.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": "你好，请回复'连接成功'"}],
            temperature=0.0
        )
        print(f"✅ Judge LLM 连接成功: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Judge LLM 连接失败: {e}")
        exit(1)
    
    print("\n" + "="*80)
    print("🧪 测试 Faithfulness 检查")
    print("="*80)
    
    test_chunks = [
        {"doc_id": "doc-1", "text": "SmartLock-100 使用 4 节 5 号电池，续航约 1 年。"},
        {"doc_id": "doc-1", "text": "SmartLock-100 支持指纹、密码、卡片、钥匙四种开锁方式。"}
    ]
    
    # 测试 1：完全忠实的答案
    test_answer_good = "SmartLock-100 使用 4 节 5 号电池，续航约 1 年，支持指纹、密码、卡片、钥匙四种开锁方式。"
    
    print("\n--- 测试忠实答案 ---")
    result = llm_faithfulness_check(test_answer_good, test_chunks, llm)
    print(result['raw_response'])
    
    # 测试 2：包含编造内容的答案
    test_answer_bad = "SmartLock-100 使用 4 节 7 号电池，续航约 2 年，支持人脸识别开锁。"
    
    print("\n--- 测试编造答案 ---")
    result = llm_faithfulness_check(test_answer_bad, test_chunks, llm)
    print(result['raw_response'])
    
    print("\n" + "="*80)
    print("🧪 测试 Answer Relevance 检查")
    print("="*80)
    
    test_query = "SmartLock-100 的电池续航时间是多久？"
    
    # 测试 1：完全相关的答案
    test_answer_relevant = "SmartLock-100 使用 4 节 5 号电池，续航约 1 年。"
    
    print("\n--- 测试相关答案 ---")
    result = llm_relevance_check(test_query, test_answer_relevant, llm)
    print(result['raw_response'])
    
    # 测试 2：不相关的答案
    test_answer_irrelevant = "SmartLock-100 支持指纹、密码、卡片、钥匙四种开锁方式。"
    
    print("\n--- 测试不相关答案 ---")
    result = llm_relevance_check(test_query, test_answer_irrelevant, llm)
    print(result['raw_response'])
    
    # 测试 3：部分相关的答案
    test_answer_partial = "SmartLock-100 使用 4 节 5 号电池，支持多种开锁方式。"
    
    print("\n--- 测试部分相关答案 ---")
    result = llm_relevance_check(test_query, test_answer_partial, llm)
    print(result['raw_response'])
    
    print("\n✅ 所有测试完成！")
