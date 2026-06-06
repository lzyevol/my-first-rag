"""
LangGraph RAG Agent — 用 StateGraph 将检索生成流程映射为图结构。

图结构（含条件分支 + 循环重试）:

                        START
                          │
                          ▼
                  ┌───────────────┐
                  │ rewrite_query │  查询重写（LLM 消解指代/补全省略）
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │   retrieve    │  混合检索（向量 + BM25 + RRF 融合）
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────────┐
                  │  check_retrieval  │  检索质量判断（计算 RRF 平均分）
                  └────┬─────────┬────┘
                       │          │
            分数 >= 阈值         分数 < 阈值
                       │          │
                       ▼          ▼
              ┌────────────┐  ┌──────────────────┐
              │  generate  │  │generate_not_found│  直接返回"找不到"
              └─────┬──────┘  └────────┬─────────┘
                    │                  │
                    ▼                  │
              ┌────────────┐           │
              │ self_check │  LLM 自检是否引用文档
              └──┬─────┬───┘           │
                 │     │               │
          引用正确   未引用 &           │
                 │  retry < max        │
                 │     │               │
                 │     └──→ generate   │
                 │            (重试)    │
                 ▼                     │
          ┌──────────────┐             │
          │update_memory │◄────────────┘
          └──────┬───────┘
                 │
                 ▼
                END

与原有线性流程的对比：
- 原流程 (main.py)：顺序调用 retrieve() → generate()，无分支无重试
- 新流程：有向图 + 条件分支 + 循环重试
- 条件边：检索质量不够 → 不走生成，直接告知用户
- 循环边：生成后自检未引用文档 → 回退重生成（最多 2 次）
"""

from typing import TypedDict, List, Optional, Literal, Annotated
from langgraph.graph import StateGraph, START, END

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_deepseek import ChatDeepSeek
from rank_bm25 import BM25Okapi

from rag_modules.retrieval import (
    bm25_search,
    vector_search,
    rrf_fusion,
    route_query,
)
from rag_modules.generation import build_context
from rag_modules.memory import ConversationMemory
from rag_modules.query_rewriter import QueryRewriter


# ═══════════════════════════════════════════════════════════════════════════════
# 图状态定义
# ═══════════════════════════════════════════════════════════════════════════════

class RAGState(TypedDict, total=False):
    """
    LangGraph 状态 — 在各节点之间流转的数据。

    字段说明：
    - query: 用户原始输入
    - rewritten_query: 查询重写后的版本
    - conversation_history: 格式化历史对话文本
    - retrieved_docs: 检索到的文档列表
    - retrieval_score: RRF 融合后的平均相关性分数（用于质量判断）
    - context: 组装好的上下文字符串
    - answer: 最终生成的回答
    - query_type: 查询类型分类
    - retry_count: 当前已重试次数（self_check fail → +1）
    - max_retries: 最大允许重试次数
    - error: 错误信息（如有）
    - self_check_result: 自检结果——""未检查 / "pass"通过 / "fail"未通过，由路由函数消费后由 generate 清除
    """
    query: str
    rewritten_query: str
    conversation_history: str
    retrieved_docs: List[Document]
    retrieval_score: float
    context: str
    answer: str
    query_type: str
    retry_count: int
    max_retries: int
    error: Optional[str]
    self_check_result: str  # "" (未检查) | "pass" | "fail" — 显式三态，替代隐式布尔标记


# ═══════════════════════════════════════════════════════════════════════════════
# 图节点函数
# ═══════════════════════════════════════════════════════════════════════════════

def make_rewrite_query_node(query_rewriter: QueryRewriter):
    """查询重写节点工厂。"""

    def rewrite_query_node(state: RAGState) -> dict:
        query = state.get("query", "").strip()
        history = state.get("conversation_history", "")

        if not query:
            return {"rewritten_query": query, "error": "查询为空"}

        rewritten = query_rewriter.rewrite(query, history_text=history)

        if rewritten != query:
            print(f"  [查询重写] '{query}' → '{rewritten}'")
        else:
            print(f"  [查询重写] 查询已清晰，无需改写")

        return {"rewritten_query": rewritten}

    return rewrite_query_node


def make_retrieve_node(
    vectorstore: FAISS,
    all_chunks: List[Document],
    bm25_index: BM25Okapi,
    top_k: int = 5,
    bm25_top_k: int = 5,
    final_top_k: int = 3,
    rrf_k: int = 60,
):
    """检索节点工厂。"""

    def retrieve_node(state: RAGState) -> dict:
        query = state.get("rewritten_query", state.get("query", ""))
        if not query:
            return {"retrieved_docs": [], "query_type": "general", "retrieval_score": 0.0}

        query_type = route_query(query)

        # BM25 关键词检索
        bm25_results = bm25_search(query, all_chunks, bm25_index, top_k=bm25_top_k)

        # FAISS 向量检索
        vector_results = vector_search(query, vectorstore, top_k=top_k)

        # RRF 融合 — 同时计算平均 RRF 分数
        final_docs, avg_rrf_score = rrf_fusion_with_score(
            bm25_results, vector_results, k=rrf_k, final_top_k=final_top_k
        )

        # 兜底
        if not final_docs:
            final_docs = bm25_results[:final_top_k]
            avg_rrf_score = 0.0

        print(f"  [检索] query_type={query_type}, vector={len(vector_results)}条, "
              f"BM25={len(bm25_results)}条, final={len(final_docs)}条, "
              f"avg_rrf_score={avg_rrf_score:.4f}")

        return {
            "retrieved_docs": final_docs,
            "query_type": query_type,
            "retrieval_score": avg_rrf_score,
        }

    return retrieve_node


# ── RRF 融合（带分数返回） ──

def rrf_fusion_with_score(
    bm25_results: List[Document],
    vector_results: List[Document],
    k: int = 60,
    final_top_k: int = 3,
):
    """
    RRF 融合 + 返回平均分数。

    与 retrieval.py 中的 rrf_fusion 逻辑相同，但额外计算并返回
    final_top_k 文档的平均 RRF 分数，供 check_retrieval 节点判断质量。
    """
    # 去重 key：优先用 (来源文件, 内容前100字符)
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

    # 按 RRF 分数降序排列
    seen = set()
    merged = []
    for doc in bm25_results + vector_results:
        doc_id = _dedup_key(doc)
        if doc_id not in seen:
            seen.add(doc_id)
            merged.append((doc, rrf_scores.get(doc_id, 0.0)))

    merged.sort(key=lambda x: x[1], reverse=True)

    # 取 top_k 并计算平均分
    top_items = merged[:final_top_k]
    avg_score = sum(s for _, s in top_items) / len(top_items) if top_items else 0.0

    # 父子文档替换
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


# ── 检索质量判断（条件边） ──

def make_check_retrieval_node(score_threshold: float = 0.01):
    """
    检索质量判断节点工厂。

    判断逻辑：
    - 如果 retrieved_docs 为空 → 直接判定为"不相关"
    - 如果 avg_rrf_score < threshold → 判定为检索质量不足
    - 否则 → 进入生成节点

    阈值说明：
    RRF 分数 = 1/(rank + k)，k=60 时，排名第1的文档得分≈0.0164
    排第3的文档得分≈0.0159。threshold=0.01 意味着要求至少有
    一个文档排在前100（即检索结果中至少有一个确实相关的）。
    实际调优中可根据业务需求调整。
    """

    def check_retrieval_node(state: RAGState) -> dict:
        docs = state.get("retrieved_docs", [])
        score = state.get("retrieval_score", 0.0)

        if not docs or len(docs) == 0:
            print(f"  [质量检查] 无检索结果 → 路由到 generate_not_found")
            return {"error": "no_results"}

        if score < score_threshold:
            print(f"  [质量检查] RRF 平均分 {score:.4f} < 阈值 {score_threshold} → 路由到 generate_not_found")
            return {"error": "low_quality"}

        print(f"  [质量检查] RRF 平均分 {score:.4f} >= 阈值 {score_threshold} → 路由到 generate")
        return {}

    return check_retrieval_node


def route_after_check(state: RAGState) -> Literal["generate", "generate_not_found"]:
    """
    条件边：根据检索质量决定路由。

    返回 "generate_not_found" 的情况：
    - state["error"] == "no_results"（完全没有检索结果）
    - state["error"] == "low_quality"（检索分数低于阈值）
    """
    error = state.get("error", "")
    if error in ("no_results", "low_quality"):
        return "generate_not_found"
    return "generate"


# ── "找不到" 生成节点 ──

def make_generate_not_found_node():
    """
    当检索质量不足时，生成礼貌的"找不到"回复。
    """

    def generate_not_found_node(state: RAGState) -> dict:
        query = state.get("query", "")
        error_type = state.get("error", "low_quality")
        print(f"  [not_found] 检索质量不足以生成可靠回答 (原因: {error_type})")

        # 简单语言检测：若 query 中 ASCII 字符占比 > 80% 且包含英文单词，用英文回复
        ascii_chars = sum(1 for c in query if ord(c) < 128)
        is_english = (
            len(query) > 0
            and ascii_chars / len(query) > 0.8
            and any(c.isalpha() for c in query)
        )

        if is_english:
            answer = (
                f"Sorry, I couldn't find any information related to \"{query}\" in my notes.\n\n"
                "Suggestions:\n"
                "1. Try rephrasing your question with different keywords\n"
                "2. Check if your notes contain relevant content\n"
                "3. Type /history to review the conversation context"
            )
        else:
            answer = (
                f"抱歉，我在笔记中没有找到与「{query}」相关的信息。\n\n"
                "建议：\n"
                "1. 尝试用不同的关键词重新提问\n"
                "2. 检查笔记中是否包含相关内容\n"
                "3. 输入 /history 查看之前的对话上下文"
            )
        return {"answer": answer, "context": ""}

    return generate_not_found_node


# ── 生成节点 ──

def make_generate_node(
    llm: ChatDeepSeek,
    memory: ConversationMemory,
    system_prompt: str,
):
    """生成节点工厂。"""

    GENERATE_USER_TEMPLATE = """{history_section}

## 参考资料（你必须基于以下资料回答，并在回答中引用来源编号）
{context}

## 当前问题
{question}

重要：回答中必须至少引用一条参考资料（如"根据[来源1]..."）。如果资料不足以回答，请如实说明。"""

    def generate_node(state: RAGState) -> dict:
        from langchain_core.prompts import ChatPromptTemplate

        query = state.get("rewritten_query", state.get("query", ""))
        docs = state.get("retrieved_docs", [])
        query_type = state.get("query_type", "general")
        retry_count = state.get("retry_count", 0)

        if not docs:
            answer = "抱歉，没有在笔记中找到与您问题相关的内容。请尝试用不同的关键词提问。"
            return {"answer": answer, "context": ""}

        retrieval_context = build_context(docs)

        # 对话历史（短期记忆）
        history_section = ""
        if not memory.is_empty:
            history_text = memory.format_for_context(max_turns=4)
            history_section = f"## 对话历史（供参考）\n{history_text}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", GENERATE_USER_TEMPLATE),
        ])

        messages = prompt.format_messages(
            history_section=history_section,
            context=retrieval_context,
            question=query,
        )

        try:
            response = llm.invoke(messages)
            answer = response.content
        except Exception as e:
            answer = f"生成回答时出错: {e}"

        retry_tag = f"(重试{retry_count})" if retry_count > 0 else ""
        print(f"  [生成{retry_tag}] query_type={query_type}, "
              f"context_chars={len(retrieval_context)}, answer_chars={len(answer)}")

        return {
            "answer": answer,
            "context": retrieval_context,
            "self_check_result": "",  # 清除上一轮自检结果，等待新一轮检查
        }

    return generate_node


# ── 自检节点（循环边） ──

SELF_CHECK_SYSTEM_PROMPT = """你是一个回答质量检查员。你的任务是检查助手回答是否引用了参考资料。

规则：
1. 检查回答中是否包含对参考资料的引用（如"根据[来源1]"、"参考资料显示"等）
2. 检查回答中的关键事实是否能在参考资料中找到对应内容
3. 如果回答完全脱离参考资料、凭空编造，标记为 "fail"
4. 如果回答基于参考资料，标记为 "pass"
5. 只输出 "pass" 或 "fail"，不要输出其他内容"""

SELF_CHECK_USER_TEMPLATE = """## 参考资料
{context}

## 助手回答
{answer}

请判断回答是否基于参考资料。只输出 pass 或 fail:"""


def make_self_check_node(llm: ChatDeepSeek, max_retries: int = 2):
    """
    自检节点工厂。

    职责：
    用 LLM 检查生成的回答是否引用了检索到的文档。
    如果未引用且重试次数未达上限 → 回到 generate 节点重新生成
    如果引用正确或重试次数已满 → 进入 update_memory
    """

    def self_check_node(state: RAGState) -> dict:
        from langchain_core.prompts import ChatPromptTemplate

        answer = state.get("answer", "")
        context = state.get("context", "")
        retry_count = state.get("retry_count", 0)

        # 如果没有上下文（比如 not_found 路径），直接通过
        if not context:
            return {"self_check_result": "pass"}

        print(f"  [自检] 第{retry_count + 1}次检查...")

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", SELF_CHECK_SYSTEM_PROMPT),
                ("human", SELF_CHECK_USER_TEMPLATE),
            ])
            messages = prompt.format_messages(context=context, answer=answer)
            response = llm.invoke(messages)
            verdict = response.content.strip().lower()
        except Exception as e:
            print(f"  [自检] 检查出错: {e}，默认通过")
            return {"self_check_result": "pass"}

        if "fail" in verdict and retry_count < max_retries:
            print(f"  [自检] ✗ 回答未引用参考资料 → 将重试 (第{retry_count + 1}次, 上限{max_retries})")
            return {
                "retry_count": retry_count + 1,
                "self_check_result": "fail",
            }
        else:
            if "pass" in verdict:
                print(f"  [自检] ✓ 回答正确引用了参考资料")
            elif retry_count >= max_retries:
                print(f"  [自检] ⚠ 已达最大重试次数 {max_retries}，强制通过")
            else:
                print(f"  [自检] 无法判断 ({verdict[:50]})，默认通过")
            return {"self_check_result": "pass"}

    return self_check_node


# ═══════════════════════════════════════════════════════════════════════════════
# 图构建与编译
# ═══════════════════════════════════════════════════════════════════════════════

class LangGraphRAGAgent:
    """
    基于 LangGraph StateGraph 的 RAG Agent（含条件分支 + 循环重试）。

    图结构:
    START → rewrite → retrieve → check_retrieval
                                      │
                          ┌───────────┼───────────┐
                          │ (分数够)              │ (分数不够)
                          ▼                       ▼
                      generate            generate_not_found
                          │                       │
                          ▼                       │
                      self_check                  │
                     ┌────┴────┐                  │
                     │ (pass)  │ (fail & retry)   │
                     ▼         ▼                  │
                update_memory  generate           │
                     │         (loop)             │
                     └─────────┴──────────────────┘
                                  │
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
        score_threshold: float = 0.01,
        max_retries: int = 2,
        system_prompt: Optional[str] = None,
    ):
        self.memory = memory or ConversationMemory()
        self.query_rewriter = query_rewriter or QueryRewriter(llm)
        self.max_retries = max_retries

        _system_prompt = system_prompt or (
            "你是我的个人知识库助手。请根据下面提供的参考资料回答问题。\n"
            "规则：\n"
            "1. 回答必须基于提供的参考资料，不要使用外部知识。\n"
            "2. 如果参考资料不足以回答问题，请直接告知：'抱歉，我的笔记中没有找到相关信息。'\n"
            "3. 回答要简洁、结构化，尽量列出要点。\n"
            "4. 如果对话历史中有相关信息，可以参考以保持回答一致性。\n"
            "5. 回答中必须明确引用参考资料的来源编号（如'根据[来源1]'）。"
        )

        # 创建节点函数
        rewrite_node = make_rewrite_query_node(self.query_rewriter)
        retrieve_node = make_retrieve_node(
            vectorstore, all_chunks, bm25_index,
            top_k=retrieval_top_k, bm25_top_k=bm25_top_k,
            final_top_k=final_top_k, rrf_k=rrf_k,
        )
        check_retrieval_node = make_check_retrieval_node(score_threshold)
        generate_node = make_generate_node(llm, self.memory, _system_prompt)
        generate_not_found_node = make_generate_not_found_node()
        self_check_node = make_self_check_node(llm, max_retries)
        update_memory_node = make_update_memory_node(self.memory)

        # 构建图
        self.graph = self._build_graph(
            rewrite_node, retrieve_node, check_retrieval_node,
            generate_node, generate_not_found_node,
            self_check_node, update_memory_node,
        )

    def _build_graph(
        self,
        rewrite_node, retrieve_node, check_retrieval_node,
        generate_node, generate_not_found_node,
        self_check_node, update_memory_node,
    ):
        """构建完整 StateGraph（含条件分支 + 循环边）。"""
        workflow = StateGraph(RAGState)

        # 添加所有节点
        workflow.add_node("rewrite_query", rewrite_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("check_retrieval", check_retrieval_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("generate_not_found", generate_not_found_node)
        workflow.add_node("self_check", self_check_node)
        workflow.add_node("update_memory", update_memory_node)

        # ── 线性部分 ──
        workflow.add_edge(START, "rewrite_query")
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("retrieve", "check_retrieval")

        # ── 条件分支 1：检索质量判断 ──
        workflow.add_conditional_edges(
            "check_retrieval",
            route_after_check,
            {
                "generate": "generate",
                "generate_not_found": "generate_not_found",
            },
        )

        # generate_not_found 直接到 update_memory
        workflow.add_edge("generate_not_found", "update_memory")

        # ── 条件分支 2 + 循环边：生成后自检 ──
        workflow.add_edge("generate", "self_check")
        workflow.add_conditional_edges(
            "self_check",
            self._route_after_self_check,
            {
                "generate": "generate",       # 循环：回去重生成
                "update_memory": "update_memory",  # 通过：更新记忆
            },
        )

        # 终点
        workflow.add_edge("update_memory", END)

        return workflow.compile()

    def _route_after_self_check(self, state: RAGState) -> Literal["generate", "update_memory"]:
        """
        条件边 2：自检后的路由逻辑。

        设计说明：
        self_check_result 是一个显式三态字段（"" / "pass" / "fail"），
        由 self_check 节点写入，由本路由函数消费，由 generate 节点清除。

        相比旧版 _should_retry 布尔标记的优势：
        1. 三态语义明确："" = 未检查, "pass" = 通过, "fail" = 未通过
        2. generate 节点清除为 "" → 避免"标记泄漏"（旧 flag 被下一轮误读）
        3. 名称 self_check_result 直接表明"是谁设置的"，降低维护者的认知负担
        """
        verdict = state.get("self_check_result", "")
        retry_count = state.get("retry_count", 0)

        if verdict == "fail" and retry_count < self.max_retries:
            return "generate"
        return "update_memory"

    def run(self, query: str) -> dict:
        """运行 RAG 管道。"""
        initial_state: RAGState = {
            "query": query,
            "rewritten_query": query,
            "conversation_history": self.memory.format_for_rewrite(),
            "retrieved_docs": [],
            "retrieval_score": 0.0,
            "context": "",
            "answer": "",
            "query_type": "general",
            "retry_count": 0,
            "max_retries": self.max_retries,
            "self_check_result": "",
        }

        result = self.graph.invoke(initial_state)
        return result

    def stream(self, query: str):
        """流式执行 RAG 管道。"""
        initial_state: RAGState = {
            "query": query,
            "rewritten_query": query,
            "conversation_history": self.memory.format_for_rewrite(),
            "retrieved_docs": [],
            "retrieval_score": 0.0,
            "context": "",
            "answer": "",
            "query_type": "general",
            "retry_count": 0,
            "max_retries": self.max_retries,
            "self_check_result": "",
        }

        for step in self.graph.stream(initial_state):
            yield step

    def reset_memory(self):
        """重置对话记忆。"""
        self.memory.clear()
        print("[记忆] 对话历史已清空")

    def get_graph_visualization(self) -> str:
        """返回 Mermaid 可视化代码。"""
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception:
            return """```mermaid
graph TD
    START --> rewrite_query
    rewrite_query --> retrieve
    retrieve --> check_retrieval
    check_retrieval -- score>=threshold --> generate
    check_retrieval -- score<threshold --> generate_not_found
    generate --> self_check
    self_check -- pass --> update_memory
    self_check -- fail&retry<max --> generate
    generate_not_found --> update_memory
    update_memory --> END
```"""

    def print_graph_structure(self):
        """打印图结构的文本表示。"""
        print("""
╔══════════════════════════════════════════════════════════╗
║     LangGraph RAG Agent — 图结构（含条件分支+循环）       ║
╠══════════════════════════════════════════════════════════╣
║                                                        ║
║   START                                                ║
║     │                                                  ║
║     ▼                                                  ║
║  ┌─────────────────┐  查询重写（LLM消解指代）           ║
║  │  rewrite_query  │                                   ║
║  └────────┬────────┘                                   ║
║           │                                            ║
║           ▼                                            ║
║  ┌─────────────────┐  混合检索（向量+BM25+RRF）         ║
║  │    retrieve     │                                   ║
║  └────────┬────────┘                                   ║
║           │                                            ║
║           ▼                                            ║
║  ┌──────────────────────┐  检索质量判断（RRF分数阈值）    ║
║  │  check_retrieval     │                              ║
║  └───┬─────────────┬────┘                              ║
║      │             │                                    ║
║  分数>=阈值      分数<阈值                               ║
║      │             │                                    ║
║      ▼             ▼                                    ║
║  ┌─────────┐  ┌──────────────────┐                      ║
║  │generate │  │generate_not_found│                      ║
║  └────┬────┘  └────────┬─────────┘                      ║
║       │                │                                ║
║       ▼                │                                ║
║  ┌────────────┐        │                                ║
║  │ self_check │ LLM自检│                                ║
║  └──┬─────┬───┘        │                                ║
║     │     │            │                                ║
║   pass  fail&retry<2   │                                ║
║     │     │            │                                ║
║     │     └──→ generate (循环重试)                       ║
║     │                  │                                ║
║     ▼                  ▼                                ║
║  ┌─────────────────────────┐                            ║
║  │    update_memory        │  更新短期记忆                ║
║  └───────────┬─────────────┘                            ║
║              │                                          ║
║              ▼                                          ║
║             END                                         ║
║                                                        ║
║  ★ 条件边：检索质量不够 → 不走生成，直接告知             ║
║  ★ 循环边：生成后自检未引用文档 → 回退重生成（最多2次）   ║
╚══════════════════════════════════════════════════════════╝
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# 记忆更新节点
# ═══════════════════════════════════════════════════════════════════════════════

def make_update_memory_node(memory: ConversationMemory):
    """记忆更新节点工厂。"""

    def update_memory_node(state: RAGState) -> dict:
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
