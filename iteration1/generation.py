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

# 尝试导入 OpenAI SDK（DeepSeek 兼容 OpenAI API 格式）
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# 系统提示词：限定模型只能基于给定资料作答
SYSTEM_PROMPT = (
    "你是一个只能根据给定资料回答问题的助手。"
    "只使用下面提供的资料片段作答，禁止使用资料之外的知识或猜测。"
    "如果资料片段中没有足够信息回答问题，必须回答\"未找到充分依据\"，不要编造。"
    "回答时请在句末用[文档X:片段N]标注引用的片段来源，X是文档编号，N是该文档内的片段序号。"
)

# 全局变量：延迟初始化 OpenAI 客户端（单例模式）
_openai_client = None


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


def _get_openai_client():
    """延迟加载 OpenAI 客户端（单例模式）
    
    避免每次调用 generate_answer 时都创建新的客户端，
    防止 httpx 客户端被关闭导致的错误。
    
    返回:
        OpenAI 客户端实例或 None（如果不可用）
    """
    global _openai_client
    
    if _openai_client is None and _OPENAI_AVAILABLE and os.environ.get("DEEPSEEK_API_KEY"):
        _openai_client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    
    return _openai_client


def generate_answer(query: str, retrieved_chunks: list, model: str = "deepseek-chat") -> str:
    """根据检索到的上下文片段生成答案
    
    参数:
        query: 用户查询问题
        retrieved_chunks: 检索到的相关文档块列表
        model: 使用的模型名称，默认 "deepseek-chat"
    
    返回:
        生成的答案文本
    """
    # 如果没有检索到任何片段，直接返回
    if not retrieved_chunks:
        return "未找到充分依据"

    # 获取 OpenAI 客户端（单例）
    client = _get_openai_client()
    
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
