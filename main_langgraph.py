"""
main_langgraph.py — 基于 LangGraph 重构的 RAG Agent 主入口。

新特性（相比 main.py）：
1. LangGraph StateGraph 图结构编排（替代线性顺序调用）
2. 查询重写：检索前用 LLM 改写模糊查询（含指代消解）
3. 短期记忆：维护多轮对话历史，支持追问和上下文引用

用法: python main_langgraph.py
启动后进入交互式问答循环，输入 quit 退出。
支持命令:
  /reset  清空对话记忆
  /graph  显示图结构
  /history 显示对话历史
  /stream <问题>  流式执行并显示每步状态
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
from rag_modules.langgraph_rag import LangGraphRAGAgent


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║    个人知识库 RAG 问答 Agent  (LangGraph 重构版)              ║
║                                                              ║
║  新特性:                                                     ║
║  ◆ LangGraph StateGraph 图结构编排                            ║
║  ◆ 查询重写 — LLM 自动消解指代、补全省略                      ║
║  ◆ 短期记忆 — 支持多轮对话追问                                ║
║                                                              ║
║  命令:                                                       ║
║  /reset   清空对话记忆                                       ║
║  /graph   显示图结构                                         ║
║  /history 显示对话历史                                       ║
║  /stream  流式执行（调试用）                                  ║
║  quit     退出                                               ║
╚══════════════════════════════════════════════════════════════╝
""")


def handle_command(cmd: str, agent: LangGraphRAGAgent) -> bool:
    """
    处理特殊命令。
    返回 True 表示应继续，False 表示退出。
    """
    cmd = cmd.strip().lower()

    if cmd in ("quit", "exit", "q"):
        print("再见！")
        return False

    if cmd == "/reset":
        agent.reset_memory()
        return True

    if cmd == "/graph":
        agent.print_graph_structure()
        return True

    if cmd == "/history":
        history = agent.memory.format_for_context()
        print(f"\n{history}\n")
        return True

    if cmd.startswith("/stream "):
        query = cmd[len("/stream "):].strip()
        if not query:
            print("请提供问题，如: /stream 番茄炒蛋怎么做？")
            return True
        print(f"\n{'='*60}")
        print(f"  流式执行: {query}")
        print(f"{'='*60}\n")
        for i, step in enumerate(agent.stream(query)):
            node_name = list(step.keys())[0]
            node_state = step[node_name]
            print(f"--- 步骤 {i+1}: [{node_name}] ---")
            # 只打印关键信息，跳过大对象
            for k, v in node_state.items():
                if k == "retrieved_docs":
                    print(f"  {k}: {len(v)} 个文档")
                elif k in ("context",) and isinstance(v, str):
                    print(f"  {k}: {len(v)} 字符")
                elif k == "answer":
                    print(f"  {k}: {v[:200]}{'...' if len(str(v)) > 200 else ''}")
                else:
                    print(f"  {k}: {v}")
            print()
        print(f"{'='*60}\n")
        return True

    # 未知命令
    if cmd.startswith("/"):
        print(f"未知命令: {cmd}")
        print("可用命令: /reset, /graph, /history, /stream <问题>")
        return True

    return True  # 普通问题返回 True，由调用方处理


def main():
    cfg = RAGConfig()

    if not cfg.llm_api_key:
        print("[X] 未设置 DEEPSEEK_API_KEY，请在 .env 文件或环境变量中配置。")
        return

    print_banner()

    # ── 1. 数据准备 ──
    print("[1/4] 数据准备...")
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
    print("\n[2/4] 索引构建...")
    vectorstore = load_or_build_index(
        chunks=chunks,
        embedding_model=cfg.embedding_model,
        embedding_device=cfg.embedding_device,
        index_dir=cfg.index_dir,
    )
    bm25_index = _build_bm25(chunks)
    print("[OK] BM25 索引就绪")

    # ── 3. 初始化 LLM ──
    print("\n[3/4] 初始化大模型...")
    llm = get_llm(
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
        api_key=cfg.llm_api_key,
    )
    print("[OK] DeepSeek 模型就绪")

    # ── 4. 初始化 LangGraph Agent ──
    print("\n[4/4] 构建 LangGraph RAG Agent...")
    memory = ConversationMemory(max_turns=10)
    query_rewriter = QueryRewriter(llm)

    agent = LangGraphRAGAgent(
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
        score_threshold=cfg.score_threshold,
        system_prompt=cfg.system_prompt,
    )
    print("[OK] LangGraph Agent 就绪")
    agent.print_graph_structure()

    # ── 交互式问答 ──
    print("一切就绪！输入问题开始对话，输入 quit 退出\n")

    while True:
        try:
            user_input = input("你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 处理命令
        is_continue = handle_command(user_input, agent)
        if not is_continue:
            break
        if user_input.startswith("/"):
            continue  # 命令已处理

        # 普通问题 → 运行 RAG 管道
        print()  # 空行分隔
        result = agent.run(user_input)

        # 输出结果（如果查询被改写，展示改写结果）
        rewritten = result.get("rewritten_query", "")
        if rewritten and rewritten != user_input:
            print(f"🔍 改写查询: {rewritten}")

        answer = result.get("answer", "[无回答]")
        print(f"\n📝 回答:\n{answer}\n")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
