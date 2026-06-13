from typing import TypedDict, List, Optional, Literal, Annotated
from langgraph.graph import StateGraph, START, END

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_deepseek import ChatDeepSeek
from rank_bm25 import BM25Okapi

from rag_modules.retrieval import (
    bm25_search,
    vector_search,
    route_query,
)
from rag_modules.generation import build_context
from rag_modules.memory import ConversationMemory
from rag_modules.query_rewriter import QueryRewriter


# ═══════════════════════════════════════════════════════════════════════════════
# 一、多 Agent 图状态定义
# ═══════════════════════════════════════════════════════════════════════════════

class MultiAgentState(TypedDict, total=False):
    """
    多 Agent RAG 的共享状态。

    所有 Agent 通过读写这个 State 来协作。
    Agent A 写入检索结果，Agent B 读取并生成答案，
    Agent B 写入反馈，Agent A 读取反馈并调整策略。

    字段分组:
    ─────────
    【用户输入】
    - query: 原始问题
    - rewritten_query: 查询重写后的版本
    - conversation_history: 对话历史（用于指代消解）

    【Router 决策】
    - route_decision: "knowledge_base" | "out_of_scope"
        knowledge_base → 进入 RAG 管道
        out_of_scope → 直接告诉用户"笔记中没有"

    【Agent A：知识获取 Agent 产出】
    - agent_a_log: Agent A 的思考日志（记录它做了什么决策）
    - retrieved_docs: 检索到的文档列表
    - retrieval_score: RRF 融合后的平均质量分数
    - acquisition_status: "success" | "low_quality" | "no_results"

    【Agent B：知识利用 Agent 产出】
    - agent_b_log: Agent B 的思考日志
    - context: 组装好的上下文字符串
    - answer: 最终生成的回答
    - utilization_status: "pass" | "needs_retry"

    【反馈循环】
    - feedback_to_acquisition: Agent B 给 Agent A 的反馈
        "你检索到的文档不相关，请尝试用 XXX 关键词重新检索"
    - retry_count: 当前重试次数
    - max_retries: 最大允许重试次数

    【元信息】
    - error: 错误信息
    - execution_trace: 完整的执行轨迹（用于调试和对比）
    """

    # 用户输入
    query: str
    rewritten_query: str
    conversation_history: str

    # Router
    route_decision: str  # "knowledge_base" | "out_of_scope"

    # Agent A: 知识获取
    agent_a_log: List[str]
    retrieved_docs: List[Document]
    retrieval_score: float
    acquisition_status: str  # "success" | "low_quality" | "no_results"

    # Agent B: 知识利用
    agent_b_log: List[str]
    context: str
    answer: str
    utilization_status: str  # "pass" | "needs_retry"

    # 反馈循环
    feedback_to_acquisition: str
    retry_count: int
    max_retries: int

    # 元信息
    query_type: str
    error: Optional[str]
    execution_trace: List[str]


# ═══════════════════════════════════════════════════════════════════════════════
# 二、Agent 角色定义（System Prompts）
# ═══════════════════════════════════════════════════════════════════════════════

# ── Router Agent ──

ROUTER_SYSTEM_PROMPT = """你是一个知识库分诊助手。你的工作是判断用户的问题是否可能在「个人学习笔记」中找到答案。

判断标准:
1. 如果问题是关于学习笔记可能涵盖的主题（如编程、AI、论文、读书笔记、方法论等），回答 "knowledge_base"
2. 如果问题是关于实时信息（天气、新闻、股价）、个人隐私、或者明显与学习笔记无关的内容，回答 "out_of_scope"
3. 如果你不确定，宁可回答 "knowledge_base"（宁可检索后说找不到，也不错杀一个可能能回答的问题）

只输出一个词: "knowledge_base" 或 "out_of_scope"，不要输出其他内容。"""

ROUTER_USER_TEMPLATE = """用户的笔记涵盖了以下主题: 编程、AI、机器学习、读书笔记、学习方法、个人知识管理。

用户问题: {query}

判断:"""

# ── Agent A: 知识获取 Agent ──

AGENT_A_SYSTEM_PROMPT = """你是一个「知识检索专家」(Knowledge Acquisition Agent)。

你的职责:
1. 分析用户问题，确定最佳检索策略
2. 改写模糊或过于简短的查询，使其更适合检索
3. 从知识库中检索最相关的文档
4. 评估检索结果的质量: 这些文档真的能回答用户的问题吗？

你不需要生成最终答案——那是 Agent B 的工作。你的价值在于找到最相关的信息。

如果 Agent B 给了你反馈（例如"检索到的文档不相关"），你必须:
1. 仔细阅读反馈，理解问题出在哪里
2. 调整检索策略: 换关键词、换检索方式、调整查询角度
3. 重新检索并返回更相关的文档

输出时，将你的检索思路记录在思考日志中。"""

AGENT_A_QUERY_REWRITE_TEMPLATE = """{feedback_section}
你是知识检索专家。请将用户问题改写为更适合检索的关键词或短句。

规则:
1. 如果问题包含指代词（它、这个、那个），结合对话历史替换为具体内容
2. 如果问题太口语化，改为更正式的检索表达
3. 提取问题中的核心概念作为检索关键词

当前任务 ——
对话历史:
{history}

用户问题: {query}
{feedback_hint}

请输出改写后的检索查询（可以是关键词组合，不一定是完整句子）:"""

# ── Agent B: 知识利用 Agent ──

AGENT_B_SYSTEM_PROMPT = """你是一个「知识利用专家」(Knowledge Utilization Agent)。

你的职责:
1. 仔细阅读 Agent A 检索到的文档
2. 基于文档内容生成准确、完整的回答
3. 在回答中明确标注引用来源
4. 自检: 你的回答是否完全基于提供的文档？有没有凭空编造的内容？

核心原则:
- 忠实于文档: 只使用提供的文档中的信息，不要使用你自己知道但文档中没有的知识
- 诚实: 如果文档不足以回答用户问题，明确告知，不要编造
- 可溯源: 每个关键事实都要标注来源编号

如果检索到的文档与用户问题完全无关，你应该:
1. 诚实告知: "抱歉，检索到的文档与您的问题不匹配"
2. 给 Agent A 发送反馈: 说明你需要什么样的信息，让 Agent A 重新检索"""

AGENT_B_GENERATE_TEMPLATE = """{history_section}

## Agent A 为你检索到的参考资料（你必须基于以下资料回答）
{context}

## 用户当前问题
{question}

## Agent A 的检索日志
{agent_a_log}

重要提醒:
1. 回答中必须至少引用一条参考资料（如"根据[来源1]..."）
2. 如果资料不足以回答，请如实说明，并写出你需要 Agent A 补充什么信息
3. 回答要简洁、结构化，列出要点"""


# ═══════════════════════════════════════════════════════════════════════════════
# 三、节点工厂函数
# ═══════════════════════════════════════════════════════════════════════════════

# ── Router 节点 ──

def make_router_node(llm: ChatDeepSeek):
    """
    Router Agent: 判断用户问题是否可能在知识库中找到答案。

    这是一个轻量级的 LLM 调用，快速二分分类。
    分类结果决定整个 RAG 管道的路由。
    """

    def router_node(state: MultiAgentState) -> dict:
        query = state.get("query", "").strip()
        if not query:
            return {"route_decision": "out_of_scope", "execution_trace": ["[Router] 查询为空，路由到 out_of_scope"]}

        # 快速关键词短路：如果问题包含这些词，直接判为 knowledge_base
        knowledge_signals = ["笔记", "学习", "论文", "代码", "算法", "编程", "AI", "RAG", "Agent",
                             "智能体", "检索", "模型", "训练", "怎么做", "步骤", "方法", "区别"]
        for signal in knowledge_signals:
            if signal in query:
                trace = [f"[Router] 检测到关键词 '{signal}' → 路由到 knowledge_base"]
                print(f"  [Router] → knowledge_base (关键词: {signal})")
                return {"route_decision": "knowledge_base", "execution_trace": trace}

        # 用 LLM 做更细致的判断
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", ROUTER_SYSTEM_PROMPT),
                ("human", ROUTER_USER_TEMPLATE),
            ])
            messages = prompt.format_messages(query=query)
            response = llm.invoke(messages)
            decision = response.content.strip().lower()
        except Exception as e:
            print(f"  [Router] LLM 调用失败: {e}，默认路由到 knowledge_base")
            return {"route_decision": "knowledge_base",
                    "execution_trace": ["[Router] LLM 失败，默认 knowledge_base"]}

        if "out_of_scope" in decision:
            trace = [f"[Router] LLM 判断为 out_of_scope"]
            print(f"  [Router] → out_of_scope")
            return {"route_decision": "out_of_scope", "execution_trace": trace}
        else:
            trace = [f"[Router] LLM 判断为 knowledge_base"]
            print(f"  [Router] → knowledge_base")
            return {"route_decision": "knowledge_base", "execution_trace": trace}

    return router_node


def route_after_router(state: MultiAgentState) -> Literal["agent_a_rewrite", "honest_response"]:
    """Router 之后的边：两个分支。"""
    decision = state.get("route_decision", "knowledge_base")
    if decision == "out_of_scope":
        return "honest_response"
    return "agent_a_rewrite"


# ── Agent A: 知识获取节点组 ──

def make_agent_a_rewrite_node(llm: ChatDeepSeek, query_rewriter: QueryRewriter):
    """
    Agent A 第一步: 查询改写。

    Agent A 收到用户问题后，先用 LLM 将问题改写为适合检索的形式。
    如果有来自 Agent B 的反馈，结合反馈调整改写策略。
    """

    def agent_a_rewrite_node(state: MultiAgentState) -> dict:
        query = state.get("query", "")
        history = state.get("conversation_history", "")
        feedback = state.get("feedback_to_acquisition", "")
        retry_count = state.get("retry_count", 0)

        log_entry = f"[Agent A] 改写查询 (重试{retry_count})"
        if feedback:
            log_entry += f" | 收到 Agent B 反馈: {feedback[:100]}"

        # 构建反馈提示
        feedback_section = ""
        feedback_hint = ""
        if feedback:
            feedback_section = "## Agent B 的反馈（上一次检索的问题）\n"
            feedback_section += f"Agent B 说上次检索的问题是: {feedback}\n"
            feedback_section += "请根据这个反馈调整查询策略。\n"
            feedback_hint = "\n（注意：上次检索结果不好，请尝试不同的关键词或角度）"

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", AGENT_A_QUERY_REWRITE_TEMPLATE),
                ("human", "{input}"),
            ])
            messages = prompt.format_messages(
                feedback_section=feedback_section,
                feedback_hint=feedback_hint,
                history=history,
                query=query,
                input=query,
            )
            response = llm.invoke(messages)
            rewritten = response.content.strip()
        except Exception:
            rewritten = query

        if not rewritten or len(rewritten) > len(query) * 3:
            rewritten = query

        if rewritten != query:
            print(f"  [Agent A-改写] '{query[:50]}...' → '{rewritten[:80]}...'")
        else:
            print(f"  [Agent A-改写] 查询已清晰，无需改写")

        log_entry += f" | 改写结果: {rewritten[:100]}"

        return {
            "rewritten_query": rewritten,
            "agent_a_log": [log_entry],
            "execution_trace": [log_entry],
        }

    return agent_a_rewrite_node


def make_agent_a_retrieve_node(
    vectorstore: FAISS,
    all_chunks: List[Document],
    bm25_index: BM25Okapi,
    top_k: int = 5,
    bm25_top_k: int = 5,
    final_top_k: int = 3,
    rrf_k: int = 60,
):
    """
    Agent A 第二步: 混合检索 + 质量评估。

    使用向量检索 + BM25 关键词检索 + RRF 融合，
    然后评估检索结果的质量。
    """

    def agent_a_retrieve_node(state: MultiAgentState) -> dict:
        query = state.get("rewritten_query", state.get("query", ""))
        if not query:
            return {
                "retrieved_docs": [],
                "retrieval_score": 0.0,
                "acquisition_status": "no_results",
                "agent_a_log": ["[Agent A] 无查询，跳过检索"],
            }

        # 查询类型判断
        query_type = route_query(query)

        # BM25 检索
        bm25_results = bm25_search(query, all_chunks, bm25_index, top_k=bm25_top_k)

        # 向量检索
        vector_results = vector_search(query, vectorstore, top_k=top_k)

        # RRF 融合 + 分数
        final_docs, avg_score = _rrf_fusion_with_score(
            bm25_results, vector_results, k=rrf_k, final_top_k=final_top_k
        )

        if not final_docs:
            final_docs = bm25_results[:final_top_k]
            avg_score = 0.0

        # 质量判断
        if not final_docs or len(final_docs) == 0:
            acquisition_status = "no_results"
        elif avg_score < 0.01:
            acquisition_status = "low_quality"
        else:
            acquisition_status = "success"

        log_entry = (
            f"[Agent A] 检索完成 | query_type={query_type} | "
            f"vector={len(vector_results)}条 BM25={len(bm25_results)}条 "
            f"final={len(final_docs)}条 | avg_rrf={avg_score:.4f} | "
            f"状态={acquisition_status}"
        )

        # 记录每个文档的主题
        doc_topics = []
        for i, doc in enumerate(final_docs[:3]):
            source = doc.metadata.get("filename", "?")
            preview = doc.page_content[:60].replace("\n", " ")
            doc_topics.append(f"  [来源{i+1}:{source}] {preview}...")

        print(f"  [Agent A-检索] {acquisition_status} | {len(final_docs)}个文档 | RRF={avg_score:.4f}")
        for topic in doc_topics:
            print(f"    {topic}")

        return {
            "retrieved_docs": final_docs,
            "retrieval_score": avg_score,
            "acquisition_status": acquisition_status,
            "query_type": query_type,
            "agent_a_log": [log_entry] + doc_topics,
            "execution_trace": [log_entry],
        }

    return agent_a_retrieve_node


def _rrf_fusion_with_score(
    bm25_results: List[Document],
    vector_results: List[Document],
    k: int = 60,
    final_top_k: int = 3,
):
    """RRF 融合 + 返回平均分数。逻辑同 retrieval.py::rrf_fusion，额外返回分数。"""
    def _dedup_key(doc: Document) -> str:
        source = doc.metadata.get("filename", doc.metadata.get("source", ""))
        content_prefix = doc.page_content[:100]
        return f"{source}::{content_prefix}"

    rrf_scores = {}
    for rank, doc in enumerate(bm25_results):
        doc_id = _dedup_key(doc)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rank + k)
    for rank, doc in enumerate(vector_results):
        doc_id = _dedup_key(doc)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rank + k)

    seen = set()
    merged = []
    for doc in bm25_results + vector_results:
        doc_id = _dedup_key(doc)
        if doc_id not in seen:
            seen.add(doc_id)
            merged.append((doc, rrf_scores.get(doc_id, 0.0)))
    merged.sort(key=lambda x: x[1], reverse=True)

    top_items = merged[:final_top_k]
    avg_score = sum(s for _, s in top_items) / len(top_items) if top_items else 0.0

    final_docs = []
    for doc, _ in top_items:
        if len(doc.page_content.strip()) < 50 and doc.metadata.get("parent_content"):
            parent_doc = Document(
                page_content=doc.metadata["parent_content"],
                metadata={**doc.metadata},
            )
            final_docs.append(parent_doc)
        else:
            final_docs.append(doc)

    return final_docs, avg_score


def route_after_acquisition(state: MultiAgentState) -> Literal["agent_b_generate", "agent_b_uncertain"]:
    """
    Agent A 检索完成后的路由:
    - 检索质量正常 → Agent B 正常生成
    - 检索质量低/无结果 → Agent B 生成不确定回答
    """
    status = state.get("acquisition_status", "success")
    if status in ("low_quality", "no_results"):
        return "agent_b_uncertain"
    return "agent_b_generate"


# ── Agent B: 知识利用节点组 ──

def make_agent_b_generate_node(llm: ChatDeepSeek, memory: ConversationMemory):
    """
    Agent B 第一步: 基于检索文档生成答案。

    Agent B 收到 Agent A 的检索结果后:
    1. 阅读检索到的文档
    2. 阅读 Agent A 的检索日志（理解检索策略）
    3. 生成基于文档的忠实答案
    """

    def agent_b_generate_node(state: MultiAgentState) -> dict:
        query = state.get("rewritten_query", state.get("query", ""))
        docs = state.get("retrieved_docs", [])
        agent_a_log = state.get("agent_a_log", [])
        retrieval_score = state.get("retrieval_score", 0.0)

        if not docs:
            answer = "抱歉，我的笔记中没有找到与您问题相关的内容。"
            return {
                "answer": answer,
                "context": "",
                "agent_b_log": ["[Agent B] 无文档，返回抱歉回答"],
                "utilization_status": "pass",
            }

        retrieval_context = build_context(docs)

        # 对话历史
        history_section = ""
        if not memory.is_empty:
            history_text = memory.format_for_context(max_turns=4)
            history_section = f"## 对话历史\n{history_text}"

        # Agent A 日志摘要
        a_log_summary = "\n".join(agent_a_log[:5]) if agent_a_log else "（无日志）"

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", AGENT_B_SYSTEM_PROMPT),
                ("human", AGENT_B_GENERATE_TEMPLATE),
            ])
            messages = prompt.format_messages(
                history_section=history_section,
                context=retrieval_context,
                question=query,
                agent_a_log=a_log_summary,
            )
            response = llm.invoke(messages)
            answer = response.content
        except Exception as e:
            answer = f"生成回答时出错: {e}"

        log_entry = (
            f"[Agent B] 生成完成 | "
            f"检索质量分={retrieval_score:.4f} | "
            f"上下文长度={len(retrieval_context)}字符 | "
            f"回答长度={len(answer)}字符"
        )
        print(f"  [Agent B-生成] 回答长度={len(answer)}字符")

        return {
            "answer": answer,
            "context": retrieval_context,
            "agent_b_log": [log_entry],
            "utilization_status": "",
            "execution_trace": [log_entry],
        }

    return agent_b_generate_node


def make_agent_b_uncertain_node(llm: ChatDeepSeek):
    """
    Agent B 变体: 当检索质量不足时，生成带有不确定性的回答。

    这个节点告诉用户"我不太确定"，而不是强行编造。
    """

    UNCERTAIN_TEMPLATE = """你是一个谨慎的知识助手。Agent A 的检索结果质量不高，请生成一个诚实的回答。

规则:
1. 如果检索到了一些文档但不太相关，尽力从中提取可能有用的信息，但明确标注"根据有限的信息"
2. 如果完全没有检索到文档，直接告知用户
3. 不要编造内容，不要假装知道

## Agent A 检索到的文档（质量可能不高）
{context}

## 用户问题
{question}

## Agent A 的检索状态
{acquisition_status}

请回答:"""

    def agent_b_uncertain_node(state: MultiAgentState) -> dict:
        query = state.get("rewritten_query", state.get("query", ""))
        docs = state.get("retrieved_docs", [])
        acquisition_status = state.get("acquisition_status", "low_quality")

        if not docs:
            answer = (
                f"抱歉，我在笔记中没有找到与「{query}」相关的信息。\n\n"
                "建议:\n"
                "1. 尝试用不同的关键词重新提问\n"
                "2. 检查笔记中是否包含相关内容\n"
                "3. 输入 /history 查看之前的对话上下文"
            )
            return {
                "answer": answer,
                "context": "",
                "agent_b_log": ["[Agent B] 无文档，返回不确定回答"],
                "utilization_status": "pass",
            }

        context = build_context(docs)

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个诚实的知识助手。请基于有限的资料尽力回答，但不要编造。"),
                ("human", UNCERTAIN_TEMPLATE),
            ])
            messages = prompt.format_messages(
                context=context,
                question=query,
                acquisition_status=acquisition_status,
            )
            response = llm.invoke(messages)
            answer = response.content
        except Exception:
            answer = "抱歉，我的笔记中似乎没有与您问题完全匹配的信息。请尝试换一种方式提问。"

        log_entry = f"[Agent B] 生成不确定回答 | 状态={acquisition_status}"
        print(f"  [Agent B-不确定] 检索质量不足，生成诚实回答")

        return {
            "answer": answer,
            "context": context,
            "agent_b_log": [log_entry],
            "utilization_status": "pass",
        }

    return agent_b_uncertain_node


# ── Agent B 自检节点（含反馈循环） ──

AGENT_B_SELF_CHECK_PROMPT = """你是 Agent B 的质量检查模块。请检查你的回答是否基于 Agent A 检索到的文档。

规则:
1. 检查回答中是否引用了参考资料（如"根据[来源1]"）
2. 检查回答中的关键事实是否能在参考资料中找到对应内容
3. 如果回答完全脱离参考资料、凭空编造 → 标记为 "needs_retry"
4. 如果回答基于参考资料 → 标记为 "pass"
5. 如果要标记为 "needs_retry"，同时写出你需要 Agent A 做什么改进

输出格式:
判定: pass 或 needs_retry
反馈: （如果是 needs_retry，写一段不超过100字的话告诉 Agent A 需要什么样的信息）"""

AGENT_B_SELF_CHECK_TEMPLATE = """## Agent A 检索到的参考资料
{context}

## 你的回答
{answer}

请自检:"""


def make_agent_b_self_check_node(llm: ChatDeepSeek, max_retries: int = 2):
    """
    Agent B 自检节点。

    核心职责:
    1. 用 LLM 检查回答是否忠实于检索文档
    2. 如果未忠实引用且重试次数未达上限 → 生成反馈给 Agent A → 回到 Agent A 重搜
    3. 如果引用正确或重试次数已满 → 进入 update_memory
    """

    def agent_b_self_check_node(state: MultiAgentState) -> dict:
        answer = state.get("answer", "")
        context = state.get("context", "")
        retry_count = state.get("retry_count", 0)

        if not context:
            print(f"  [Agent B-自检] 无上下文，直接通过")
            return {"utilization_status": "pass"}

        print(f"  [Agent B-自检] 第{retry_count + 1}次检查...")

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", AGENT_B_SELF_CHECK_PROMPT),
                ("human", AGENT_B_SELF_CHECK_TEMPLATE),
            ])
            messages = prompt.format_messages(context=context[:3000], answer=answer[:1500])
            response = llm.invoke(messages)
            verdict_text = response.content.strip()
        except Exception as e:
            print(f"  [Agent B-自检] LLM 调用失败: {e}，默认通过")
            return {"utilization_status": "pass"}

        # 解析判定和反馈
        is_needs_retry = "needs_retry" in verdict_text.lower()
        feedback = ""
        if "反馈:" in verdict_text:
            feedback = verdict_text.split("反馈:", 1)[1].strip()[:200]
        elif "反馈：" in verdict_text:
            feedback = verdict_text.split("反馈：", 1)[1].strip()[:200]

        if is_needs_retry and retry_count < max_retries:
            print(f"  [Agent B-自检] ✗ 回答未基于文档 → 发送反馈给 Agent A (第{retry_count + 1}次)")
            print(f"    反馈: {feedback[:100]}...")
            return {
                "utilization_status": "needs_retry",
                "feedback_to_acquisition": feedback,
                "retry_count": retry_count + 1,
                "agent_b_log": [f"[Agent B] 自检失败 → 反馈给 Agent A: {feedback[:100]}"],
                "execution_trace": [f"[Agent B] 自检失败，发送反馈(重试{retry_count + 1})"],
            }
        else:
            if is_needs_retry:
                print(f"  [Agent B-自检] ⚠ 已达最大重试次数 {max_retries}，强制通过")
            else:
                print(f"  [Agent B-自检] ✓ 回答正确引用了参考资料")
            return {"utilization_status": "pass"}

    return agent_b_self_check_node


def route_after_self_check(state: MultiAgentState) -> Literal["agent_a_rewrite", "update_memory"]:
    """
    自检后的路由:
    - 需要重试 → 回到 Agent A 重搜
    - 通过 → 进入记忆更新 → 结束
    """
    status = state.get("utilization_status", "pass")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if status == "needs_retry" and retry_count < max_retries:
        print("  [路由] 自检未通过 → Agent A 重新检索")
        return "agent_a_rewrite"
    return "update_memory"


# ── 诚实回复节点 ──

def make_honest_response_node():
    """
    当 Router 判定问题超出知识库范围时，生成诚实的"不知道"回复。
    """

    def honest_response_node(state: MultiAgentState) -> dict:
        query = state.get("query", "")
        print(f"  [诚实回复] 问题超出知识库范围: '{query[:50]}...'")

        answer = (
            f"抱歉，我的笔记中可能没有与「{query}」相关的信息。\n\n"
            f"我的知识库主要包含: 编程、AI、读书笔记、学习方法等个人学习内容。\n"
            f"如果您的问题确实与这些主题相关，请尝试:\n"
            f"1. 用更具体的关键词重新提问\n"
            f"2. 说明您想了解的具体方面\n"
            f"3. 输入 /history 查看之前的对话上下文"
        )

        return {
            "answer": answer,
            "context": "",
            "agent_b_log": ["[诚实回复 Agent] 问题超出知识库范围"],
            "utilization_status": "pass",
            "execution_trace": ["[诚实回复] 直接返回'不知道'"],
        }

    return honest_response_node


# ── 记忆更新节点 ──

def make_update_memory_node(memory: ConversationMemory):
    """记忆更新节点——Agent B 的回答存入对话历史。"""

    def update_memory_node(state: MultiAgentState) -> dict:
        query = state.get("query", "")
        answer = state.get("answer", "")
        query_type = state.get("query_type", "general")

        if query:
            memory.add_user_query(query, query_type)
        if answer:
            memory.add_assistant_response(answer)

        print(f"  [记忆] 当前对话轮数: {memory.turn_count}")
        return {}

    return update_memory_node


# ═══════════════════════════════════════════════════════════════════════════════
# 四、多 Agent 图类
# ═══════════════════════════════════════════════════════════════════════════════

class MultiAgentRAG:
    """
    多智能体 RAG 系统。

    图结构:

    START
      │
      ▼
    ┌──────────────────┐
    │  router          │  ← Router Agent: 判断问题是否在知识库范围内
    └──┬───────────┬───┘
       │           │
       │ knowledge │ out_of_scope
       │ _base     │
       ▼           ▼
    ┌──────────┐ ┌──────────────────┐
    │agent_a_  │ │honest_response   │  ← 诚实回复 Agent
    │rewrite   │ │"我笔记中没有"    │
    └────┬─────┘ └────────┬─────────┘
         │                │
         ▼                │
    ┌──────────┐          │
    │agent_a_  │          │
    │retrieve  │  ← Agent A (知识获取)
    └────┬─────┘
         │
    ┌────┼────────┐
    │    │        │
    │ quality_ok  │ quality_low
    │    │        │
    ▼    │        ▼
    ┌────────┐ ┌──────────────┐
    │agent_b │ │agent_b       │
    │_gen    │ │_uncertain    │  ← Agent B (知识利用)
    └───┬────┘ └──────┬───────┘
        │              │
        ▼              │
    ┌──────────┐       │
    │agent_b   │       │
    │self_check│       │
    └──┬───┬───┘       │
       │   │           │
    pass  needs_retry  │
       │   │           │
       │   └──→ agent_a_rewrite  ← 反馈循环!
       │               │
       ▼               ▼
    ┌──────────────────────┐
    │  update_memory       │
    └──────────┬───────────┘
               ▼
              END
    """

    def __init__(
        self,
        vectorstore: FAISS,
        bm25_index: BM25Okapi,
        all_chunks: List[Document],
        llm: ChatDeepSeek,
        memory: Optional[ConversationMemory] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        retrieval_top_k: int = 5,
        bm25_top_k: int = 5,
        final_top_k: int = 3,
        rrf_k: int = 60,
        max_retries: int = 2,
    ):
        self.memory = memory or ConversationMemory()
        self.query_rewriter = query_rewriter or QueryRewriter(llm)
        self.max_retries = max_retries
        self.llm = llm

        # 创建节点
        router_node = make_router_node(llm)
        agent_a_rewrite = make_agent_a_rewrite_node(llm, self.query_rewriter)
        agent_a_retrieve = make_agent_a_retrieve_node(
            vectorstore, all_chunks, bm25_index,
            top_k=retrieval_top_k, bm25_top_k=bm25_top_k,
            final_top_k=final_top_k, rrf_k=rrf_k,
        )
        agent_b_generate = make_agent_b_generate_node(llm, self.memory)
        agent_b_uncertain = make_agent_b_uncertain_node(llm)
        agent_b_self_check = make_agent_b_self_check_node(llm, max_retries)
        honest_response = make_honest_response_node()
        update_memory = make_update_memory_node(self.memory)

        # 构建图
        self.graph = self._build_graph(
            router_node,
            agent_a_rewrite, agent_a_retrieve,
            agent_b_generate, agent_b_uncertain,
            agent_b_self_check,
            honest_response,
            update_memory,
        )

    def _build_graph(self, router, a_rewrite, a_retrieve, b_gen, b_uncertain,
                     b_check, honest, update_mem):
        """构建完整多 Agent 图。"""
        workflow = StateGraph(MultiAgentState)

        # 添加所有节点
        workflow.add_node("router", router)
        workflow.add_node("agent_a_rewrite", a_rewrite)
        workflow.add_node("agent_a_retrieve", a_retrieve)
        workflow.add_node("agent_b_generate", b_gen)
        workflow.add_node("agent_b_uncertain", b_uncertain)
        workflow.add_node("agent_b_self_check", b_check)
        workflow.add_node("honest_response", honest)
        workflow.add_node("update_memory", update_mem)

        # ── 图结构 ──

        # START → Router
        workflow.add_edge(START, "router")

        # Router 分支
        workflow.add_conditional_edges(
            "router",
            route_after_router,
            {
                "agent_a_rewrite": "agent_a_rewrite",
                "honest_response": "honest_response",
            },
        )

        # Agent A 线性流程: rewrite → retrieve
        workflow.add_edge("agent_a_rewrite", "agent_a_retrieve")

        # Agent A → Agent B 分支（检索质量判断）
        workflow.add_conditional_edges(
            "agent_a_retrieve",
            route_after_acquisition,
            {
                "agent_b_generate": "agent_b_generate",
                "agent_b_uncertain": "agent_b_uncertain",
            },
        )

        # Agent B 正常生成 → 自检
        workflow.add_edge("agent_b_generate", "agent_b_self_check")

        # Agent B 不确定生成 → 记忆更新（跳过自检，直接通过）
        workflow.add_edge("agent_b_uncertain", "update_memory")

        # 自检分支 → 通过/重试
        workflow.add_conditional_edges(
            "agent_b_self_check",
            route_after_self_check,
            {
                "agent_a_rewrite": "agent_a_rewrite",     # ← 反馈循环
                "update_memory": "update_memory",
            },
        )

        # 诚实回复 → 记忆更新
        workflow.add_edge("honest_response", "update_memory")

        # 记忆更新 → 结束
        workflow.add_edge("update_memory", END)

        return workflow.compile()

    def run(self, query: str) -> dict:
        """运行多 Agent RAG 管道。"""
        initial_state: MultiAgentState = {
            "query": query,
            "rewritten_query": query,
            "conversation_history": self.memory.format_for_rewrite(),
            "route_decision": "",
            "agent_a_log": [],
            "retrieved_docs": [],
            "retrieval_score": 0.0,
            "acquisition_status": "",
            "agent_b_log": [],
            "context": "",
            "answer": "",
            "utilization_status": "",
            "feedback_to_acquisition": "",
            "retry_count": 0,
            "max_retries": self.max_retries,
            "query_type": "general",
            "execution_trace": [],
        }

        result = self.graph.invoke(initial_state)
        return result

    def stream(self, query: str):
        """流式执行多 Agent RAG 管道（调试用）。"""
        initial_state: MultiAgentState = {
            "query": query,
            "rewritten_query": query,
            "conversation_history": self.memory.format_for_rewrite(),
            "route_decision": "",
            "agent_a_log": [],
            "retrieved_docs": [],
            "retrieval_score": 0.0,
            "acquisition_status": "",
            "agent_b_log": [],
            "context": "",
            "answer": "",
            "utilization_status": "",
            "feedback_to_acquisition": "",
            "retry_count": 0,
            "max_retries": self.max_retries,
            "query_type": "general",
            "execution_trace": [],
        }

        for step in self.graph.stream(initial_state):
            yield step

    def reset_memory(self):
        """重置对话记忆。"""
        self.memory.clear()
        print("[记忆] 对话历史已清空")

    def print_graph_structure(self):
        """打印多 Agent 图结构。"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║        多智能体 RAG — 图结构（双 Agent + Router + 反馈循环）  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║                        START                                 ║
║                          │                                   ║
║                          ▼                                   ║
║  ┌──────────────────────────────┐                           ║
║  │  router                      │  Router Agent (分诊台)     ║
║  │  "知识库能回答这个问题吗？"   │                           ║
║  └──┬───────────────────────┬───┘                           ║
║     │                       │                                ║
║     │ knowledge_base        │ out_of_scope                   ║
║     ▼                       ▼                                ║
║  ┌──────────────────┐  ┌──────────────────────┐             ║
║  │ agent_a_rewrite  │  │ honest_response      │             ║
║  │ (查询改写)        │  │ "抱歉，笔记中没有"   │             ║
║  └────────┬─────────┘  └──────────┬───────────┘             ║
║           │                       │                          ║
║           ▼                       │                          ║
║  ┌──────────────────┐             │                          ║
║  │ agent_a_retrieve │             │                          ║
║  │ (混合检索+质量评) │ ← Agent A  │                          ║
║  └────────┬─────────┘             │                          ║
║           │                       │                          ║
║     ┌─────┼─────┐                 │                          ║
║     │     │     │                 │                          ║
║  quality quality low               │                          ║
║    ok    │                         │                          ║
║     │     │                        │                          ║
║     ▼     ▼                        │                          ║
║  ┌──────────────┐ ┌─────────────┐  │                          ║
║  │agent_b_gen   │ │agent_b_     │  │                          ║
║  │(正常生成)     │ │uncertain    │  │                          ║
║  └──────┬───────┘ │(诚实回答)   │  │                          ║
║         │         └──────┬──────┘  │                          ║
║         ▼                │         │                          ║
║  ┌──────────────┐        │         │                          ║
║  │agent_b_self_ │ ← Agent B       │                          ║
║  │check (自检)  │        │         │                          ║
║  └──┬───────┬───┘        │         │                          ║
║     │       │            │         │                          ║
║   pass  needs_retry      │         │                          ║
║     │       │            │         │                          ║
║     │       └──→ agent_a_rewrite    │                          ║
║     │           (反馈循环!)          │                          ║
║     │                               │                          ║
║     ▼                               ▼                          ║
║  ┌──────────────────────────────────────┐                     ║
║  │  update_memory (记忆更新)             │                     ║
║  └──────────────────┬───────────────────┘                     ║
║                     ▼                                         ║
║                    END                                        ║
║                                                              ║
║  ★ Router Agent: 问题分诊，决定是否进入知识库检索              ║
║  ★ Agent A (知识获取): 负责检索+质量评估                       ║
║  ★ Agent B (知识利用): 负责生成+自检+反馈                     ║
║  ★ 反馈循环: Agent B 自检失败 → 告诉 A 哪里不好 → A 重新检索  ║
╚══════════════════════════════════════════════════════════════════╝
        """)

    def get_graph_visualization(self) -> str:
        """返回 Mermaid 可视化代码。"""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception:
            return """```mermaid
graph TD
    START --> router
    router -- knowledge_base --> agent_a_rewrite
    router -- out_of_scope --> honest_response
    agent_a_rewrite --> agent_a_retrieve
    agent_a_retrieve -- quality_ok --> agent_b_generate
    agent_a_retrieve -- quality_low --> agent_b_uncertain
    agent_b_generate --> agent_b_self_check
    agent_b_self_check -- pass --> update_memory
    agent_b_self_check -- needs_retry --> agent_a_rewrite
    agent_b_uncertain --> update_memory
    honest_response --> update_memory
    update_memory --> END
```"""
