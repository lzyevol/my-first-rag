"""
rag_modules 单元测试。

覆盖范围：
- 检索：RRF 融合、查询路由、去重 key
- LangGraph：检索质量路由、自检路由、generate_not_found 语言检测
- 记忆：滑动窗口、格式化

运行方式：python -m pytest tests/test_rag_modules.py -v
（从 my-first-rag 项目根目录执行）
"""

import pytest
import sys
import os

# 确保项目根在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助工厂函数
# ═══════════════════════════════════════════════════════════════════════════════

def make_doc(content: str, filename: str = "test.md") -> Document:
    """创建带 filename 元数据的 Document，模拟真实分块输出。"""
    return Document(page_content=content, metadata={"filename": filename})


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RRF 融合测试（retrieval.py）
# ═══════════════════════════════════════════════════════════════════════════════

class TestRRFFusion:
    """测试 rag_modules.retrieval.rrf_fusion"""

    def test_basic_fusion_returns_merged_docs(self):
        from rag_modules.retrieval import rrf_fusion

        bm25 = [make_doc("Doc A content", "a.md")]
        vec = [make_doc("Doc B content", "b.md")]

        result = rrf_fusion(bm25, vec, final_top_k=2)
        assert len(result) == 2

    def test_fusion_respects_final_top_k(self):
        from rag_modules.retrieval import rrf_fusion

        bm25 = [make_doc(f"Doc {i}") for i in range(5)]
        vec = [make_doc(f"Doc {i}") for i in range(5)]

        result = rrf_fusion(bm25, vec, final_top_k=3)
        assert len(result) == 3

    def test_duplicate_docs_are_deduped(self):
        """两边检索返回同一文档时，应去重而非重复。"""
        from rag_modules.retrieval import rrf_fusion

        same_doc = make_doc("same content here", "same.md")
        bm25 = [same_doc, make_doc("other doc", "other.md")]
        vec = [same_doc, make_doc("third doc", "third.md")]

        result = rrf_fusion(bm25, vec, final_top_k=5)
        # 去重后应有 3 个不同文档
        unique_contents = {doc.page_content for doc in result}
        assert len(unique_contents) == 3

    def test_deduplication_with_different_source_same_prefix(self):
        """不同来源但前 100 字符相同时，不应被误去重。"""
        from rag_modules.retrieval import rrf_fusion

        prefix = "A" * 100
        doc_a = make_doc(prefix + " from a", "a.md")
        doc_b = make_doc(prefix + " from b", "b.md")

        bm25 = [doc_a]
        vec = [doc_b]

        result = rrf_fusion(bm25, vec, final_top_k=5)
        # 两个不同来源的文档都应保留
        assert len(result) == 2

    def test_empty_input_returns_empty(self):
        from rag_modules.retrieval import rrf_fusion

        result = rrf_fusion([], [], final_top_k=3)
        assert result == []

    def test_parent_replacement_for_short_chunks(self):
        """子块太短且有 parent_content 时替换为父文档。"""
        from rag_modules.retrieval import rrf_fusion

        child = Document(
            page_content="short",
            metadata={"filename": "test.md", "parent_content": "Full parent content here"},
        )
        result = rrf_fusion([child], [], final_top_k=1)
        assert result[0].page_content == "Full parent content here"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 查询路由测试（retrieval.py）
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryRouting:
    """测试 rag_modules.retrieval.route_query"""

    def test_list_keywords_detect_list_type(self):
        from rag_modules.retrieval import route_query

        assert route_query("有哪些科幻书推荐？") == "list"
        assert route_query("列举所有分类") == "list"

    def test_detail_keywords_detect_detail_type(self):
        from rag_modules.retrieval import route_query

        assert route_query("番茄炒蛋怎么做？") == "detail"
        assert route_query("Python和Rust有什么区别？") == "detail"
        assert route_query("如何学习机器学习？") == "detail"

    def test_general_query_returns_general(self):
        from rag_modules.retrieval import route_query

        assert route_query("番茄炒蛋") == "general"
        assert route_query("今天天气怎么样") == "general"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RRF 融合带分数测试（langgraph_rag.py）
# ═══════════════════════════════════════════════════════════════════════════════

class TestRRFFusionWithScore:
    """测试 langgraph_rag.rrf_fusion_with_score"""

    def test_returns_docs_and_score(self):
        from rag_modules.langgraph_rag import rrf_fusion_with_score

        bm25 = [make_doc(f"Doc {i}") for i in range(3)]
        vec = [make_doc(f"Doc {i}") for i in range(3)]

        docs, score = rrf_fusion_with_score(bm25, vec, final_top_k=2)
        assert len(docs) == 2
        assert score > 0.0

    def test_empty_input_returns_zero_score(self):
        from rag_modules.langgraph_rag import rrf_fusion_with_score

        docs, score = rrf_fusion_with_score([], [], final_top_k=3)
        assert docs == []
        assert score == 0.0

    def test_score_bounded_by_rrf_formula(self):
        """RRF 分数上限：k=60 时单检索器排名第 1 得 1/60 ≈ 0.0167。"""
        from rag_modules.langgraph_rag import rrf_fusion_with_score

        docs = [make_doc(f"Doc {i}") for i in range(10)]
        _, score = rrf_fusion_with_score(docs, [], k=60, final_top_k=3)
        # 平均分不应超过理论最大值
        assert score <= 1.0 / 60


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LangGraph 路由函数测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouteAfterCheck:
    """测试 langgraph_rag.route_after_check"""

    def test_no_error_routes_to_generate(self):
        from rag_modules.langgraph_rag import route_after_check

        state = {"error": ""}
        assert route_after_check(state) == "generate"

    def test_no_results_routes_to_not_found(self):
        from rag_modules.langgraph_rag import route_after_check

        state = {"error": "no_results"}
        assert route_after_check(state) == "generate_not_found"

    def test_low_quality_routes_to_not_found(self):
        from rag_modules.langgraph_rag import route_after_check

        state = {"error": "low_quality"}
        assert route_after_check(state) == "generate_not_found"

    def test_missing_error_key_defaults_to_generate(self):
        """error 字段不存在时应安全降级到 generate。"""
        from rag_modules.langgraph_rag import route_after_check

        state = {}
        assert route_after_check(state) == "generate"


class TestRouteAfterSelfCheck:
    """测试 LangGraphRAGAgent._route_after_self_check"""

    def test_fail_with_retries_left_routes_to_generate(self):
        from rag_modules.langgraph_rag import LangGraphRAGAgent
        from unittest.mock import MagicMock

        agent = MagicMock(spec=LangGraphRAGAgent)
        agent.max_retries = 2

        state = {"self_check_result": "fail", "retry_count": 1}
        route_fn = LangGraphRAGAgent._route_after_self_check
        result = route_fn(agent, state)
        assert result == "generate"

    def test_fail_with_max_retries_routes_to_update_memory(self):
        from rag_modules.langgraph_rag import LangGraphRAGAgent
        from unittest.mock import MagicMock

        agent = MagicMock(spec=LangGraphRAGAgent)
        agent.max_retries = 2

        state = {"self_check_result": "fail", "retry_count": 2}
        route_fn = LangGraphRAGAgent._route_after_self_check
        result = route_fn(agent, state)
        assert result == "update_memory"

    def test_pass_routes_to_update_memory(self):
        from rag_modules.langgraph_rag import LangGraphRAGAgent
        from unittest.mock import MagicMock

        agent = MagicMock(spec=LangGraphRAGAgent)
        agent.max_retries = 2

        state = {"self_check_result": "pass", "retry_count": 0}
        route_fn = LangGraphRAGAgent._route_after_self_check
        result = route_fn(agent, state)
        assert result == "update_memory"

    def test_empty_result_defaults_to_update_memory(self):
        """未设置 self_check_result 时安全降级——不等，直接通过。"""
        from rag_modules.langgraph_rag import LangGraphRAGAgent
        from unittest.mock import MagicMock

        agent = MagicMock(spec=LangGraphRAGAgent)
        agent.max_retries = 2

        state = {"retry_count": 0}
        route_fn = LangGraphRAGAgent._route_after_self_check
        result = route_fn(agent, state)
        assert result == "update_memory"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. generate_not_found 语言检测测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateNotFoundLanguage:
    """测试 generate_not_found_node 的语言检测逻辑"""

    def test_chinese_query_returns_chinese(self):
        from rag_modules.langgraph_rag import make_generate_not_found_node

        node = make_generate_not_found_node()
        result = node({"query": "量子力学是什么？", "error": "low_quality"})
        assert "抱歉" in result["answer"]
        assert "Sorry" not in result["answer"]

    def test_english_query_returns_english(self):
        from rag_modules.langgraph_rag import make_generate_not_found_node

        node = make_generate_not_found_node()
        result = node({"query": "What is quantum mechanics?", "error": "low_quality"})
        assert "Sorry" in result["answer"]

    def test_mixed_query_defaults_to_chinese(self):
        """中英混合（中文占比高）时默认中文。"""
        from rag_modules.langgraph_rag import make_generate_not_found_node

        node = make_generate_not_found_node()
        result = node({"query": "什么是quantum mechanics？请解释一下", "error": "low_quality"})
        assert "抱歉" in result["answer"]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 记忆模块测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestConversationMemory:
    """测试 rag_modules.memory.ConversationMemory"""

    def test_sliding_window_enforces_max_turns(self):
        from rag_modules.memory import ConversationMemory

        mem = ConversationMemory(max_turns=4)
        for i in range(6):
            mem.add_user_query(f"Q{i}")
            mem.add_assistant_response(f"A{i}")

        # 6 轮问答 = 12 条记录，max_turns=4 应只保留最后 4 条
        assert len(mem) == 4
        # 最旧的两轮已被移除
        assert mem._history[0].content == "Q4"

    def test_is_empty(self):
        from rag_modules.memory import ConversationMemory

        mem = ConversationMemory()
        assert mem.is_empty

        mem.add_user_query("test")
        assert not mem.is_empty

    def test_turn_count_counts_queries_only(self):
        from rag_modules.memory import ConversationMemory

        mem = ConversationMemory()
        mem.add_user_query("Q1")
        mem.add_assistant_response("A1")
        mem.add_user_query("Q2")

        assert mem.turn_count == 2  # 两次用户提问

    def test_format_for_rewrite_truncates_long_assistant_responses(self):
        from rag_modules.memory import ConversationMemory

        mem = ConversationMemory()
        mem.add_user_query("test")
        mem.add_assistant_response("A" * 200)

        formatted = mem.format_for_rewrite()
        # 助手回答应被截断到 100 字符 + "..."
        assert "..." in formatted
        assert len("A" * 200 + "...") > len(formatted)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 查询重写模块测试
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryRewriterHeuristics:
    """测试 QueryRewriter._seems_clear 启发式判断"""

    def test_short_query_is_not_clear(self):
        from rag_modules.query_rewriter import QueryRewriter

        assert not QueryRewriter._seems_clear("怎么做")

    def test_query_with_pronoun_is_not_clear(self):
        from rag_modules.query_rewriter import QueryRewriter

        assert not QueryRewriter._seems_clear("它的步骤是什么？")

    def test_omission_signal_is_not_clear(self):
        from rag_modules.query_rewriter import QueryRewriter

        assert not QueryRewriter._seems_clear("还有别的推荐吗")

    def test_full_question_is_clear(self):
        from rag_modules.query_rewriter import QueryRewriter

        assert QueryRewriter._seems_clear("Python装饰器的用法是什么？")


# ═══════════════════════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
