"""
生成集成模块 — 将检索到的上下文与用户问题组装，调用 DeepSeek 生成答案。
"""
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek


SYSTEM_PROMPT = """你是一个知识渊博的助手。请根据下面的上下文和你的知识回答用户问题。
  如果上下文有帮助就参考，没有就用自己的知识回答。
  回答要详细、完整。"""

USER_TEMPLATE = """上下文:
{context}

问题: {question}

回答:"""


def build_context(docs: List[Document]) -> str:
    """将多个检索文档拼接成上下文字符串，并标注来源。"""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", "未知来源")
        parts.append(f"[来源{i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def get_llm(model: str, temperature: float, max_tokens: int, api_key: str) -> ChatDeepSeek:
    """初始化 DeepSeek 大模型。"""
    return ChatDeepSeek(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )


def generate(
    question: str,
    retrieved_docs: List[Document],
    llm: ChatDeepSeek,
    query_type: str = "general",
) -> str:
    """
    生成答案的主入口。
    query_type 用于后续扩展（如 list 用简洁模式，detail 用详细模式）。
    """
    if not retrieved_docs:
        return "抱歉，没有在笔记中找到相关内容。"

    context = build_context(retrieved_docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE),
    ])

    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    return response.content
