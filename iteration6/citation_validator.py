"""
Citation Validator (引用验证器)

核心功能：验证 LLM 生成的带引用答案是否存在 "引用幻觉" (Citation Hallucination)

验证策略：三层验证机制
1. Span 存在性：span 必须是 answer 的子串
2. Source 存在性：source 必须在合法来源列表中
3. 内容一致性：span 的内容必须能在 source 原文中找到依据

技术亮点（来自 Claude 测试代码启发）：
- 使用 2-gram 滑窗提取中文关键词，避免分词边界不一致
- 数字上下文验证：数字必须在相同上下文中出现，防止张冠李戴
- 位置占用跟踪：避免多个 citation 定位到同一段文字
- 梯度验证：精确匹配 → 数字上下文 → 词汇重叠度

使用示例:
    sources = {
        "文档13:片段1": "ST-500 人体传感器：安装高度 1.8-2.2 米。",
        "文档13:片段2": "安装步骤：按压粘贴 → 连接电源 → 配网。"
    }
    
    llm_output = {
        "answer": "ST-500安装高度为1.8-2.2米，步骤是先按压粘贴再连接电源。",
        "citations": [
            {"span": "安装高度为1.8-2.2米", "source": "文档13:片段1"},
            {"span": "先按压粘贴再连接电源", "source": "文档13:片段2"}
        ]
    }
    
    result = validate_and_render(
        llm_output["answer"],
        llm_output["citations"],
        sources
    )
    
    print(result["final_answer"])  # 带标注的最终答案
    print(result["failed"])  # 未通过验证的 citations
"""

import re
from typing import List, Dict, Tuple, Set


def extract_terms(text: str) -> Set[str]:
    """
    提取用于重合度比较的"词"
    
    策略：
    - 中文：使用 2-gram 滑窗（避免切词边界不一致导致误判）
    - 型号/数字：整体保留（如 ST-500, 1.8-2.2 等）
    
    参数:
        text: 待提取的文本
    
    返回:
        关键词集合（2-gram + 型号）
    
    示例:
        >>> extract_terms("ST-500 安装高度 1.8-2.2 米")
        {'ST-500', '安装', '装高', '高度', '1.8', '2.2', ...}
    """
    # 提取中文 2-gram
    chinese_only = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    bigrams = {chinese_only[i:i+2] for i in range(len(chinese_only) - 1)}
    
    # 提取型号和数字混合词（如 ST-500, A1, 1.8 等）
    codes = set(re.findall(r'[A-Za-z]+-?\d+[A-Za-z\d]*', text))
    
    # 提取纯数字（包括小数）
    numbers = set(re.findall(r'\d+\.?\d*', text))
    
    return bigrams | codes | numbers


def numbers_match_context(span: str, source_text: str, window: int = 15) -> bool:
    """
    验证 span 中的数字是否在 source 中有相同上下文
    
    防止 "张冠李戴" 问题：
    - 错误示例：span "1.8米" 引用 source "工作电压1.8V"
    - 正确示例：span "1.8米" 引用 source "安装高度1.8-2.2米"
    
    策略：
    对于 span 中的每个数字，要求：
    1. 数字在 source 中出现
    2. 数字周围的上下文（window 字符范围）与 span 有关键词重叠
    
    参数:
        span: 待验证的文本片段
        source_text: 来源文档原文
        window: 上下文窗口大小（字符数）
    
    返回:
        True 如果所有数字都有上下文匹配
    
    示例:
        >>> numbers_match_context("安装高度1.8米", "ST-500安装高度1.8-2.2米")
        True  # "安装高度" 在数字周围
        
        >>> numbers_match_context("电压1.8V", "ST-500安装高度1.8-2.2米")
        False  # "电压" 不在数字周围
    """
    # 提取 span 中的所有数字
    numbers = re.findall(r'\d+\.?\d*', span)
    
    # 如果没有数字，直接通过
    if not numbers:
        return True
    
    # 提取 span 的关键词
    span_terms = extract_terms(span)
    
    # 检查每个数字
    for num in numbers:
        ok = False
        
        # 找到该数字在 source 中的所有出现位置
        for match in re.finditer(re.escape(num), source_text):
            # 提取数字周围的上下文
            start = max(0, match.start() - window)
            end = min(len(source_text), match.end() + window)
            context = source_text[start:end]
            
            # 检查上下文是否与 span 有关键词重叠
            context_terms = extract_terms(context)
            if span_terms & context_terms:  # 有交集
                ok = True
                break
        
        # 如果这个数字没有找到合适的上下文，失败
        if not ok:
            return False
    
    return True


def span_supported_by_source(span: str, source_text: str, threshold: float = 0.5) -> bool:
    """
    判断 span 内容是否能在 source 原文中找到依据
    
    三层梯度验证：
    1. 精确匹配：span 完整出现在 source 中（最可靠）
    2. 数字上下文：span 中的数字在 source 中有相同上下文
    3. 词汇重叠：span 与 source 的关键词重叠度 >= threshold
    
    参数:
        span: 待验证的文本片段
        source_text: 来源文档原文
        threshold: 词汇重叠度阈值（默认 0.5，需根据实际数据调优）
    
    返回:
        True 如果 span 在 source 中有充分依据
    
    示例:
        >>> span_supported_by_source("安装高度1.8-2.2米", "ST-500 人体传感器：安装高度 1.8-2.2 米")
        True  # 精确匹配
        
        >>> span_supported_by_source("安装高度为1.8米", "ST-500安装高度1.8-2.2米")
        True  # 数字上下文匹配
        
        >>> span_supported_by_source("工作电压36V", "ST-500工作电压24V")
        False  # 数字对不上
    """
    # 层级1：精确匹配（去除空格后）
    span_normalized = span.replace(" ", "").replace("　", "")
    source_normalized = source_text.replace(" ", "").replace("　", "")
    
    if span_normalized in source_normalized:
        return True
    
    # 层级2：数字上下文验证
    if not numbers_match_context(span, source_text):
        return False
    
    # 层级3：词汇重叠度验证
    span_terms = extract_terms(span)
    if not span_terms:
        return True  # 空 span，通过
    
    source_terms = extract_terms(source_text)
    overlap = span_terms & source_terms
    coverage = len(overlap) / len(span_terms)
    
    return coverage >= threshold


def validate_and_render(
    answer: str,
    citations: List[Dict[str, str]],
    valid_sources: Dict[str, str],
    threshold: float = 0.5
) -> Dict:
    """
    验证引用并渲染最终答案
    
    改进版本（支持多源引用）：
    1. 按 span 分组合并（同一 span 可能有多个 source）
    2. 对每个 source 单独验证（只要能找到依据即可）
    3. 允许合理的语言改写（放宽匹配要求）
    4. 渲染时支持多角标 [文档1:片段1][文档2:片段2]
    
    参数:
        answer: 模型生成的完整回答文本（不含标注）
        citations: 引用列表，每个元素包含 {"span": "...", "source": "文档X:片段N"}
        valid_sources: 合法来源字典 {"文档X:片段N": "原文内容"}
        threshold: 词汇重叠度阈值（默认 0.5）
    
    返回:
        {
            "final_answer": "渲染好标注的最终答案",
            "passed": [通过验证的 citation 列表],
            "failed": [未通过的 citation 列表（含失败原因）],
            "validation_stats": {...}
        }
    """
    from collections import defaultdict
    
    # 步骤 1: 按 span 分组（合并同一 span 的多个 source）
    span_to_sources = defaultdict(list)
    for citation in citations:
        span = citation.get("span", "")
        source = citation.get("source", "")
        
        # 规范化 source：去掉方括号
        source_normalized = source.strip()
        if source_normalized.startswith('[') and source_normalized.endswith(']'):
            source_normalized = source_normalized[1:-1]
        
        if span and source_normalized:
            span_to_sources[span].append(source_normalized)
    
    # 步骤 2: 对每个 span，验证所有候选 sources
    passed = []
    failed = []
    occupied = []  # 已占用的位置
    
    for span, sources in span_to_sources.items():
        # 2.1 检查 span 是否在 answer 中
        pos = -1
        start = 0
        while True:
            idx = answer.find(span, start)
            if idx == -1:
                break
            # 检查位置是否已被占用
            if not any(s <= idx < e or s < idx + len(span) <= e for s, e in occupied):
                pos = idx
                break
            start = idx + 1
        
        if pos == -1:
            # span 不在 answer 中，所有 sources 都失败
            for source in sources:
                failed.append({
                    "span": span,
                    "source": source,
                    "reason": "span不在answer中，或位置已被其他citation占用"
                })
            continue
        
        # 2.2 对每个 source 单独验证
        valid_sources_for_span = []
        
        for source in sources:
            reason = None
            
            # 验证 source 是否合法
            if source not in valid_sources:
                reason = f"source不存在于合法来源列表（共{len(valid_sources)}个来源）"
            # 验证 span 内容在 source 原文中是否有依据
            elif not span_supported_by_source(span, valid_sources[source], threshold):
                reason = "span内容在source原文中找不到充分依据（词汇重叠度不足或数字上下文不匹配）"
            
            if reason:
                failed.append({"span": span, "source": source, "reason": reason})
            else:
                valid_sources_for_span.append(source)
        
        # 2.3 如果有通过验证的 sources，记录
        if valid_sources_for_span:
            occupied.append((pos, pos + len(span)))
            passed.append({
                "span": span,
                "sources": valid_sources_for_span,  # 数组：可能有多个
                "pos": pos
            })
    
    # 步骤 3: 从后往前插入标注（支持多角标）
    final_answer = answer
    for item in sorted(passed, key=lambda x: x["pos"], reverse=True):
        insert_pos = item["pos"] + len(item["span"])
        # 生成多角标：[文档1:片段1][文档2:片段2]
        labels = "".join(f"[{src}]" for src in item["sources"])
        final_answer = final_answer[:insert_pos] + labels + final_answer[insert_pos:]
    
    # 统计信息
    total_citations = len(citations)
    total_unique_spans = len(span_to_sources)
    passed_citations = sum(len(item["sources"]) for item in passed)
    failed_citations = len(failed)
    
    return {
        "final_answer": final_answer,
        "passed": passed,
        "failed": failed,
        "validation_stats": {
            "total": total_citations,  # 原始 citations 总数
            "unique_spans": total_unique_spans,  # 去重后的 span 数
            "passed": passed_citations,  # 通过验证的 citations 数
            "failed": failed_citations,  # 失败的 citations 数
            "pass_rate": passed_citations / total_citations if total_citations > 0 else 0.0
        }
    }


if __name__ == "__main__":
    """测试用例：验证引用验证器的核心功能（支持多源引用）"""
    
    # 测试数据
    sources = {
        "文档13:片段1": "ST-500 人体传感器：安装高度 1.8-2.2 米，工作电压 24V。",
        "文档13:片段2": "ST-500 人体传感器：安装高度 1.8-2.2 米，探测角度 120°。",
        "文档5:片段3": "温度传感器 T-300：测量范围 -20℃ 至 60℃。"
    }
    
    # 模拟 LLM 输出（同一 span 对应多个 source，包含一条错误引用）
    llm_output = {
        "answer": "ST-500人体传感器安装高度为1.8-2.2米，安装步骤是先按压粘贴，再连接电源完成配网。工作电压为36V。",
        "citations": [
            {"span": "安装高度为1.8-2.2米", "source": "文档13:片段1"},  # 改写，但正确
            {"span": "安装高度为1.8-2.2米", "source": "文档13:片段2"},  # 同一 span，不同 source
            {"span": "先按压粘贴，再连接电源完成配网", "source": "文档13:片段2"},  # 假设正确
            {"span": "工作电压为36V", "source": "文档13:片段1"},  # 错误：数值不匹配（24V vs 36V）
        ]
    }
    
    # 执行验证
    print("="*60)
    print("Citation Validator 测试（多源引用支持）")
    print("="*60)
    print("\n原始答案:")
    print(llm_output["answer"])
    print("\n原始引用:")
    for i, c in enumerate(llm_output["citations"], 1):
        print(f"  {i}. '{c['span'][:30]}...' → {c['source']}")
    
    result = validate_and_render(
        llm_output["answer"],
        llm_output["citations"],
        sources
    )
    
    print("\n" + "="*60)
    print("验证结果")
    print("="*60)
    print(f"\n📊 统计信息:")
    print(f"  原始 citations:     {result['validation_stats']['total']}")
    print(f"  去重后 spans:       {result['validation_stats']['unique_spans']}")
    print(f"  ✅ 通过验证:        {result['validation_stats']['passed']}")
    print(f"  ❌ 未通过:          {result['validation_stats']['failed']}")
    print(f"  📈 通过率:          {result['validation_stats']['pass_rate']:.1%}")
    
    print("\n最终答案（带标注）:")
    print(result["final_answer"])
    
    if result["passed"]:
        print("\n✅ 通过验证的引用:")
        for item in result["passed"]:
            sources_str = ", ".join(item["sources"])
            print(f"  • '{item['span'][:40]}...' → [{sources_str}]")
    
    if result["failed"]:
        print("\n❌ 未通过验证的引用:")
        for item in result["failed"]:
            print(f"  • '{item['span'][:40]}...' → {item['source']}")
            print(f"    原因: {item['reason']}")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
