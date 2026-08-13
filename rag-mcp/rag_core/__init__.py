"""
RAG Core Module

核心 RAG 功能，独立于 MCP 协议层。
"""

from .search import search_knowledge
from .generate import generate_answer, generate_answer_async, generate_answer_with_retrieval

__all__ = [
    'search_knowledge',
    'generate_answer',
    'generate_answer_async',
    'generate_answer_with_retrieval'
]
