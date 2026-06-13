"""
main_multi_agent.py — 多智能体 RAG Agent 主入口。

运行方式:
  python main_multi_agent.py         # 多 Agent 模式（默认）
  python main_multi_agent.py --compare   # 对比模式：同时运行单 Agent 和多 Agent

特性:
  1. Router Agent: 问题分诊，判断是否在知识库范围内
  2. Agent A (知识获取): 查询改写 + 混合检索 + 质量评估
  3. Agent B (知识利用): 生成答案 + 自检 + 向 Agent A 发送反馈
  4. 反馈循环: Agent B 发现检索不行 → Agent A 重新检索
  5. 诚实回复: 知识库范围外的问题直接告知用户

命令:
  /reset    清空对话记忆
  /graph    显示多 Agent 图结构
  /history  显示对话历史
  /trace    显示上次执行轨迹
  /stream   流式执行（调试用）
  /compare  对比单 Agent vs 多 Agent 回答
  quit      退出
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RAGConfig
from rag_modules.data_preparation import prepare_data
from rag_modules.index_construction import load_or_build_index
from rag_modules.retrieval import _build_bm25
from rag_modules.generation import get_llm
from rag_modules.memory import ConversationMemory
from rag_modules.query_rewriter import QueryRewriter
from rag_modules.multi_agent_rag import MultiAgentRAG
from rag_modules.langgraph_rag import LangGraphRAGAgent


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║      个人知识库 RAG 问答 Agent — 多智能体版                   ║
║                                                              ║
║  架构:                                                       ║
║  ◆ Router Agent — 问题分诊台                                 ║
║  ◆ Agent A (知识获取) — 检索专家                             ║
║  ◆ Agent B (知识利用) — 生成+自检专家                        ║
║  ◆ 反馈循环 — Agent B → Agent A 协作重试                     ║
║                                                              ║
║  命令:                                                       ║
║  /reset    清空对话记忆                                      ║
║  /graph    显示图结构                                        ║
║  /history  显示对话历史                                      ║
║  /trace    显示上次执行轨迹                                  ║
║  /stream   流式执行（调试用）                                 ║
║  /compare  对比单 Agent vs 多 Agent 回答                      ║
║  quit      退出                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def handle_command(cmd: str, agent: MultiAgentRAG, last_trace: list, old_agent=None) -> tuple:
    """
    处理特殊命令。
    返回 (should_continue: bool, last_trace: list)
    """
    cmd = cmd.strip().lower()

    if cmd in ("quit", "exit", "q"):
        print("再见！")
        return False, last_trace

    if cmd == "/reset":
        agent.reset_memory()
        return True, last_trace

    if cmd == "/graph":
        agent.print_graph_structure()
        return True, last_trace

    if cmd == "/history":
        history = agent.memory.format_for_context()
        print(f"\n{history}\n")
        return True, last_trace

    if cmd == "/trace":
        if not last_trace:
            print("（暂无执行轨迹，请先提问）")
        else:
            print("\n## 上次执行轨迹\n")
            for i, step in enumerate(last_trace, 1):
                print(f"  {i}. {step}")
            print()
        return True, last_trace

    if cmd.startswith("/stream "):
        query = cmd[len("/stream "):].strip()
        if not query:
            print("请提供问题，如: /stream 番茄炒蛋怎么做？")
            return True, last_trace
        print(f"\n{'='*60}")
        print(f"  流式执行: {query}")
        print(f"{'='*60}\n")
        trace = []
        for i, step in enumerate(agent.stream(query)):
            node_name = list(step.keys())[0]
            node_state = step[node_name]
            trace.append(f"[{node_name}]")
            print(f"--- 步骤 {i+1}: [{node_name}] ---")
            for k, v in node_state.items():
                if k == "retrieved_docs":
                    print(f"  {k}: {len(v)} 个文档")
                elif k in ("context",) and isinstance(v, str):
                    print(f"  {k}: {len(v)} 字符")
                elif k == "answer":
                    print(f"  {k}: {v[:200]}{'...' if len(str(v)) > 200 else ''}")
                elif k in ("agent_a_log", "agent_b_log", "execution_trace"):
                    print(f"  {k}: {len(v)} 条记录")
                elif k == "route_decision":
                    print(f"  {k}: {v}")
                elif k == "acquisition_status":
                    print(f"  {k}: {v}")
                elif k == "utilization_status":
                    print(f"  {k}: {v}")
                elif k == "feedback_to_acquisition":
                    if v:
                        print(f"  {k}: {str(v)[:100]}")
                else:
                    val_str = str(v)
                    if len(val_str) < 80:
                        print(f"  {k}: {v}")
                    else:
                        print(f"  {k}: {val_str[:80]}...")
            print()
        print(f"{'='*60}\n")
        return True, trace

    if cmd.startswith("/compare "):
        query = cmd[len("/compare "):].strip()
        if not query:
            print("请提供问题，如: /compare 什么是RAG？")
            return True, last_trace
        if old_agent is None:
            print("对比模式需要同时初始化单 Agent 版本，请使用 --compare 参数启动。")
            return True, last_trace
        _run_comparison(query, agent, old_agent)
        return True, last_trace

    # 未知命令
    if cmd.startswith("/"):
        print(f"未知命令: {cmd}")
        print("可用命令: /reset, /graph, /history, /trace, /stream, /compare")
        return True, last_trace

    return True, last_trace  # 普通问题


def _run_comparison(query: str, multi_agent: MultiAgentRAG, single_agent: LangGraphRAGAgent):
    """运行单 Agent vs 多 Agent 对比。"""
    print(f"\n{'='*70}")
    print(f"  对比模式: 单 Agent vs 多 Agent")
    print(f"  问题: {query}")
    print(f"{'='*70}")

    # 多 Agent
    print("\n--- 多 Agent 回答 ---")
    multi_result = multi_agent.run(query)
    multi_answer = multi_result.get("answer", "[无回答]")
    multi_trace = multi_result.get("execution_trace", [])
    print(f"\n{multi_answer}")

    # 单 Agent（需要单独运行内存隔离的实例）
    print(f"\n--- 单 Agent 回答 ---")
    single_result = single_agent.run(query)
    single_answer = single_result.get("answer", "[无回答]")
    print(f"\n{single_answer}")

    # 对比摘要
    print(f"\n{'─'*70}")
    print("  对比摘要")
    print(f"{'─'*70}")
    print(f"  多 Agent 执行步数: {len(multi_trace)} 步")
    print(f"  多 Agent 回答长度: {len(multi_answer)} 字符")
    print(f"  单 Agent 回答长度: {len(single_answer)} 字符")
    print(f"\n  💡 对比要点:")
    print(f"     1. 哪个回答更准确？")
    print(f"     2. 哪个回答更忠实于原文？")
    print(f"     3. 多 Agent 的路由是否正确？")
    print(f"     4. 如果两者不同，为什么？")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="多智能体 RAG Agent")
    parser.add_argument("--compare", action="store_true",
                        help="同时启动单 Agent 版本用于对比")
    args = parser.parse_args()

    cfg = RAGConfig()

    if not cfg.llm_api_key:
        print("[X] 未设置 DEEPSEEK_API_KEY，请在 .env 文件或环境变量中配置。")
        return

    print_banner()

    # ── 1. 数据准备 ──
    print("[1/5] 数据准备...")
    raw_docs, chunks = prepare_data(
        data_dir=cfg.data_dir,
        pattern=cfg.glob_pattern,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    if not chunks:
        print("请将 Markdown 笔记放入 'notes/' 文件夹后重新运行。")
        return

    # ── 2. 索引构建 ──
    print("\n[2/5] 索引构建...")
    vectorstore = load_or_build_index(
        chunks=chunks,
        embedding_model=cfg.embedding_model,
        embedding_device=cfg.embedding_device,
        index_dir=cfg.index_dir,
    )
    bm25_index = _build_bm25(chunks)
    print("[OK] BM25 索引就绪")

    # ── 3. 初始化 LLM ──
    print("\n[3/5] 初始化大模型...")
    llm = get_llm(
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
        api_key=cfg.llm_api_key,
    )
    print("[OK] DeepSeek 模型就绪")

    # ── 4. 初始化多 Agent 系统 ──
    print("\n[4/5] 构建多智能体 RAG 系统...")
    memory = ConversationMemory(max_turns=10)
    query_rewriter = QueryRewriter(llm)

    multi_agent = MultiAgentRAG(
        vectorstore=vectorstore,
        bm25_index=bm25_index,
        all_chunks=chunks,
        llm=llm,
        memory=memory,
        query_rewriter=query_rewriter,
        retrieval_top_k=cfg.retrieval_top_k,
        bm25_top_k=cfg.bm25_top_k,
        final_top_k=cfg.final_top_k,
        rrf_k=cfg.rrf_k,
        max_retries=2,
    )
    print("[OK] 多智能体 RAG 系统就绪")

    # ── 5. (可选) 初始化单 Agent 版本用于对比 ──
    single_agent = None
    if args.compare:
        print("\n[5/5] 初始化单 Agent 版本（用于对比）...")
        single_memory = ConversationMemory(max_turns=10)
        single_rewriter = QueryRewriter(llm)
        single_agent = LangGraphRAGAgent(
            vectorstore=vectorstore,
            bm25_index=bm25_index,
            all_chunks=chunks,
            llm=llm,
            memory=single_memory,
            query_rewriter=single_rewriter,
            retrieval_top_k=cfg.retrieval_top_k,
            bm25_top_k=cfg.bm25_top_k,
            final_top_k=cfg.final_top_k,
            rrf_k=cfg.rrf_k,
            score_threshold=cfg.score_threshold,
            system_prompt=cfg.system_prompt,
        )
        print("[OK] 单 Agent 版本就绪（内存隔离）")
    else:
        print("\n[5/5] 跳过（使用 --compare 参数启动对比模式）")

    # 打印图结构
    multi_agent.print_graph_structure()

    # ── 交互式问答 ──
    print("一切就绪！输入问题开始对话，输入 quit 退出\n")
    print("💡 试试问: '我的笔记里有哪些学习内容？' 或者 '什么是RAG？'\n")

    last_trace = []

    while True:
        try:
            user_input = input("你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 处理命令
        is_continue, last_trace = handle_command(user_input, multi_agent, last_trace, single_agent)
        if not is_continue:
            break
        if user_input.startswith("/"):
            continue

        # 普通问题 → 多 Agent 管道
        print()
        result = multi_agent.run(user_input)

        # 输出执行轨迹摘要
        trace = result.get("execution_trace", [])
        last_trace = trace
        route = result.get("route_decision", "?")
        acq_status = result.get("acquisition_status", "?")

        print(f"\n📊 执行轨迹:")
        for t in trace:
            print(f"  {t}")

        # 输出改写结果
        rewritten = result.get("rewritten_query", "")
        if rewritten and rewritten != user_input:
            print(f"🔍 改写查询: {rewritten}")

        # 输出回答
        answer = result.get("answer", "[无回答]")
        print(f"\n📝 回答:\n{answer}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
