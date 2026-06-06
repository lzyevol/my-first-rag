"""
生成集成模块 — 将检索到的上下文与用户问题组装，调用 DeepSeek 生成答案。
"""
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek


SYSTEM_PROMPT = """你是我的个人知识库助手。请根据下面提供的参考资料回答问题。
规则：
1. 回答必须基于提供的参考资料，不要使用外部知识。
2. 如果参考资料不足以回答问题，请直接告知：'抱歉，我的笔记中没有找到相关信息。'
3. 回答要详细、完整，尽量结构化地列出要点。
4. 回答中应引用参考资料的来源编号（如'根据[来源1]'）。"""

USER_TEMPLATE = """## 参考资料
{context}

## 问题
{question}

## 回答（请基于以上资料，并标注来源编号）:"""


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

    Args:
        question: 用户问题
        retrieved_docs: 检索到的文档列表
        llm: 大模型实例
        query_type: 查询类型 — "list" 列举类 / "detail" 步骤类 / "general" 通用
    """
    if not retrieved_docs:
        return "抱歉，没有在笔记中找到相关内容。"

    context = build_context(retrieved_docs)

    # 根据查询类型附加生成风格提示
    type_hints = {
        "list": "\n（请用列表形式组织回答，如 1. 2. 3.）",
        "detail": "\n（请给出详细的步骤说明，每一步都要解释清楚。）",
        "general": "",
    }
    question_with_hint = question + type_hints.get(query_type, "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE),
    ])

    messages = prompt.format_messages(
        context=context,
        question=question_with_hint,
    )
    response = llm.invoke(messages)
    return response.content
