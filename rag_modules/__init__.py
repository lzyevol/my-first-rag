"""
rag_modules — 个人知识库 RAG 问答 Agent 核心模块。

公开 API：
- 原版（线性管道）：retrieve, generate, build_context, get_llm
- LangGraph 版（图结构）：LangGraphRAGAgent
- 辅助模块：ConversationMemory, QueryRewriter
"""

from rag_modules.retrieval import retrieve, vector_search, bm25_search, rrf_fusion, route_query, _build_bm25
from rag_modules.generation import generate, build_context, get_llm
from rag_modules.memory import ConversationMemory
from rag_modules.query_rewriter import QueryRewriter
from rag_modules.langgraph_rag import LangGraphRAGAgent

__all__ = [
    # 检索
    "retrieve",
    "vector_search",
    "bm25_search",
    "rrf_fusion",
    "route_query",
    "_build_bm25",
    # 生成
    "generate",
    "build_context",
    "get_llm",
    # 记忆
    "ConversationMemory",
    # 查询重写
    "QueryRewriter",
    # LangGraph Agent
    "LangGraphRAGAgent",
]
