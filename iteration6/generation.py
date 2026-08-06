"""
生成模块：使用 DeepSeek API 根据检索到的上下文片段生成答案

如果 DEEPSEEK_API_KEY 环境变量已设置且 openai 包已安装，则使用 DeepSeek API；
否则回退到模拟模式，这样整个流程（检索 -> 生成 -> 评分）仍可端到端运行，
无需网络访问或 API 密钥。

这个模拟回退是真实的设计选择，不只是开发便利性考虑：
Iteration 0 的验收标准是"流程端到端运行"，而不是"生成质量好"——
因此它不应该被缺失的凭证所阻塞。

Iteration 6 新增：
- 结构化 JSON 输出（带 span-based citations）
- 三层引用验证机制（span 存在性、source 合法性、内容一致性）
- 从后往前插入引用标注（避免位置偏移）
"""
import os
import asyncio
from typing import Dict, List

# 尝试导入 OpenAI SDK（DeepSeek 兼容 OpenAI API 格式）
try:
    from openai import OpenAI, AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# Iteration 6: 结构化输出提示词（支持引用验证）
SYSTEM_PROMPT_V6 = """你是一个只能根据给定资料回答问题的助手。

**核心规则**：
1. 只使用下面提供的资料片段作答，禁止使用资料之外的知识或猜测
2. 如果资料片段中没有足够信息回答问题，answer 字段填写"未找到充分依据"，citations 为空数组
3. 必须以 JSON 格式输出，包含两个字段：answer 和 citations

**输出格式要求**：
```json
{
  "answer": "完整的回答文本（不含引用标注）",
  "citations": [
    {"span": "必须是answer的完整子串", "source": "必须是资料中的标签"},
    {"span": "另一个子串", "source": "对应的标签"}
  ]
}
```

**Citations 字段说明**：
- span: 必须是 answer 中的**完整连续子串**（逐字匹配）
- source: 必须是资料中真实存在的标签（如 [文档13:片段1]），**不要编造标签**
- 每个 span 应该尽量保留原文关键词，便于后续验证
- 如果某句话综合了多个片段，为每个事实分别标注来源

**重要**：只输出 JSON，不要输出其他任何文字或解释。
"""

# 兼容旧版本的系统提示词（Iteration 5 及之前）
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


def _build_context_v6(retrieved_chunks):
    """
    Iteration 6 版本：构建上下文 + 列出所有合法标签
    
    增强功能：
    - 明确列出所有可用的引用标签，防止 LLM 编造标签
    - 提供标签到原文的映射（用于后续验证）
    
    参数:
        retrieved_chunks: 检索到的文档块列表
    
    返回:
        (context_str, valid_labels, chunks_map)
        - context_str: 格式化的上下文字符串
        - valid_labels: 所有合法标签列表 ["文档13:片段1", ...]
        - chunks_map: 标签到原文的映射 {"文档13:片段1": "原文内容"}
    """
    lines = []
    valid_labels = []
    chunks_map = {}
    
    for i, c in enumerate(retrieved_chunks):
        doc_num = c["doc_id"].replace("doc", "")
        label = f"文档{doc_num}:片段{i+1}"
        content = c['text']
        
        lines.append(f"[{label}] {content}")
        valid_labels.append(label)
        chunks_map[label] = content
    
    context_str = "\n\n".join(lines)
    return context_str, valid_labels, chunks_map


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


def generate_answer_v6(
    query: str, 
    retrieved_chunks: list, 
    model: str = "deepseek-chat", 
    client=None,
    enable_validation: bool = True,
    validation_threshold: float = 0.5
) -> Dict:
    """
    Iteration 6: 带引用验证的生成函数
    
    工作流程:
    1. 构建上下文 + 提取所有合法标签
    2. 使用结构化提示词要求 LLM 输出 JSON {"answer": "...", "citations": [...]}
    3. 解析 JSON 并进行三层引用验证
    4. 渲染最终答案（从后往前插入标注）
    
    参数:
        query: 用户查询问题
        retrieved_chunks: 检索到的相关文档块列表
        model: 使用的模型名称，默认 "deepseek-chat"
        client: OpenAI 客户端（如果为 None，会自动创建）
        enable_validation: 是否启用引用验证（默认 True）
        validation_threshold: 词汇重叠度阈值（默认 0.5）
    
    返回:
        {
            "answer": "最终答案（带引用标注）",
            "raw_answer": "原始答案（不含标注）",
            "citations": [...],  # 所有 citations（包括通过和未通过的）
            "validation": {
                "enabled": bool,
                "passed": [...],  # 通过验证的 citations
                "failed": [...],  # 未通过的 citations（含失败原因）
                "stats": {...}  # 统计信息
            },
            "llm_raw_response": "LLM 原始响应"
        }
    
    示例:
        >>> result = generate_answer_v6("ST-500 安装高度？", chunks)
        >>> print(result["answer"])
        'ST-500安装高度为1.8-2.2米[文档13:片段1]'
        >>> print(result["validation"]["stats"]["pass_rate"])
        1.0
    """
    # 如果没有检索到任何片段，直接返回
    if not retrieved_chunks:
        return {
            "answer": "未找到充分依据",
            "raw_answer": "未找到充分依据",
            "citations": [],
            "validation": {
                "enabled": enable_validation,
                "passed": [],
                "failed": [],
                "stats": {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
            },
            "llm_raw_response": ""
        }
    
    # 如果没有提供客户端，尝试创建
    if client is None:
        client = get_generator_client()
    
    # 如果客户端不可用，回退到 mock
    if not client:
        return {
            "answer": _mock_generate(query, retrieved_chunks),
            "raw_answer": _mock_generate(query, retrieved_chunks),
            "citations": [],
            "validation": {
                "enabled": False,
                "passed": [],
                "failed": [],
                "stats": {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
            },
            "llm_raw_response": ""
        }
    
    # 构建上下文 + 提取合法标签
    context, valid_labels, chunks_map = _build_context_v6(retrieved_chunks)
    
    # 构建提示词（明确列出所有合法标签）
    labels_list = ", ".join(f"[{label}]" for label in valid_labels)
    prompt = f"""资料：
{context}

**可用的引用标签（请只使用这些标签，不要编造）**：
{labels_list}

问题：{query}

请按照系统提示的 JSON 格式回答。"""
    
    # 调用 LLM（使用 JSON 模式）
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=800,
            response_format={"type": "json_object"},  # 强制 JSON 输出
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_V6},
                {"role": "user", "content": prompt}
            ],
        )
        
        raw_response = resp.choices[0].message.content
        
        # 解析 JSON
        import json
        llm_output = json.loads(raw_response)
        
        raw_answer = llm_output.get("answer", "")
        citations = llm_output.get("citations", [])
        
    except Exception as e:
        # JSON 解析失败，返回错误信息
        return {
            "answer": f"[ERROR] JSON 解析失败: {e}",
            "raw_answer": "",
            "citations": [],
            "validation": {
                "enabled": enable_validation,
                "passed": [],
                "failed": [{"span": "", "source": "", "reason": f"JSON解析失败: {e}"}],
                "stats": {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
            },
            "llm_raw_response": raw_response if 'raw_response' in locals() else ""
        }
    
    # 如果启用验证，进行引用验证
    if enable_validation:
        from citation_validator import validate_and_render
        
        validation_result = validate_and_render(
            raw_answer,
            citations,
            chunks_map,
            threshold=validation_threshold
        )
        
        final_answer = validation_result["final_answer"]
        validation_info = {
            "enabled": True,
            "passed": validation_result["passed"],
            "failed": validation_result["failed"],
            "stats": validation_result["validation_stats"]
        }
    else:
        # 不验证，直接使用原始答案
        final_answer = raw_answer
        validation_info = {
            "enabled": False,
            "passed": [],
            "failed": [],
            "stats": {"total": len(citations), "passed": 0, "failed": 0, "pass_rate": 0.0}
        }
    
    return {
        "answer": final_answer,
        "raw_answer": raw_answer,
        "citations": citations,
        "validation": validation_info,
        "llm_raw_response": raw_response
    }


async def generate_answer_v6_async(
    query: str,
    retrieved_chunks: list,
    model: str = "deepseek-chat",
    client=None,
    enable_validation: bool = True,
    validation_threshold: float = 0.5
) -> Dict:
    """
    Iteration 6: 带引用验证的生成函数（异步版本）
    
    用于批量生成，提升评估速度
    
    参数和返回值同 generate_answer_v6
    """
    # 如果没有检索到任何片段，直接返回
    if not retrieved_chunks:
        return {
            "answer": "未找到充分依据",
            "raw_answer": "未找到充分依据",
            "citations": [],
            "validation": {
                "enabled": enable_validation,
                "passed": [],
                "failed": [],
                "stats": {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
            },
            "llm_raw_response": ""
        }
    
    # 如果没有提供客户端，尝试创建
    if client is None:
        if not _OPENAI_AVAILABLE:
            return {
                "answer": _mock_generate(query, retrieved_chunks),
                "raw_answer": _mock_generate(query, retrieved_chunks),
                "citations": [],
                "validation": {
                    "enabled": False,
                    "passed": [],
                    "failed": [],
                    "stats": {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
                },
                "llm_raw_response": ""
            }
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return {
                "answer": _mock_generate(query, retrieved_chunks),
                "raw_answer": _mock_generate(query, retrieved_chunks),
                "citations": [],
                "validation": {
                    "enabled": False,
                    "passed": [],
                    "failed": [],
                    "stats": {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
                },
                "llm_raw_response": ""
            }
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    
    # 构建上下文 + 提取合法标签
    context, valid_labels, chunks_map = _build_context_v6(retrieved_chunks)
    
    # 构建提示词（明确列出所有合法标签）
    labels_list = ", ".join(f"[{label}]" for label in valid_labels)
    prompt = f"""资料：
{context}

**可用的引用标签（请只使用这些标签，不要编造）**：
{labels_list}

问题：{query}

请按照系统提示的 JSON 格式回答。"""
    
    # 调用 LLM（使用 JSON 模式）
    try:
        resp = await client.chat.completions.create(
            model=model,
            max_tokens=800,
            response_format={"type": "json_object"},  # 强制 JSON 输出
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_V6},
                {"role": "user", "content": prompt}
            ],
        )
        
        raw_response = resp.choices[0].message.content
        
        # 解析 JSON
        import json
        llm_output = json.loads(raw_response)
        
        raw_answer = llm_output.get("answer", "")
        citations = llm_output.get("citations", [])
        
    except Exception as e:
        # JSON 解析失败，返回错误信息
        return {
            "answer": f"[ERROR] JSON 解析失败: {e}",
            "raw_answer": "",
            "citations": [],
            "validation": {
                "enabled": enable_validation,
                "passed": [],
                "failed": [{"span": "", "source": "", "reason": f"JSON解析失败: {e}"}],
                "stats": {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
            },
            "llm_raw_response": raw_response if 'raw_response' in locals() else ""
        }
    
    # 如果启用验证，进行引用验证
    if enable_validation:
        from citation_validator import validate_and_render
        
        validation_result = validate_and_render(
            raw_answer,
            citations,
            chunks_map,
            threshold=validation_threshold
        )
        
        final_answer = validation_result["final_answer"]
        validation_info = {
            "enabled": True,
            "passed": validation_result["passed"],
            "failed": validation_result["failed"],
            "stats": validation_result["validation_stats"]
        }
    else:
        # 不验证，直接使用原始答案
        final_answer = raw_answer
        validation_info = {
            "enabled": False,
            "passed": [],
            "failed": [],
            "stats": {"total": len(citations), "passed": 0, "failed": 0, "pass_rate": 0.0}
        }
    
    return {
        "answer": final_answer,
        "raw_answer": raw_answer,
        "citations": citations,
        "validation": validation_info,
        "llm_raw_response": raw_response
    }
