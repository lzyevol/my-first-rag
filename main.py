"""RAG 的完整流程是什么
main.py -- 个人知识库 RAG 问答 Agent 主入口。
用法:python main.py
启动后进入交互式问答循环，输入 quit 退出。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RAGConfig    
#  从 config.py 里导入 RAGConfig 类，后面用它读取所有设置
from rag_modules.data_preparation import prepare_data
# 从 rag_modules/data_preparation.py 里导入 prepare_data 函数，负责加载笔记和切块
from rag_modules.index_construction import load_or_build_index
#导入索引管理函数。load_or_build_index = "有索引就加载，没有就新建"。
from rag_modules.retrieval import retrieve, _build_bm25
#
from rag_modules.generation import generate, get_llm


def main():
    cfg = RAGConfig()

    if not cfg.llm_api_key:
        print("[X] 未设置 DEEPSEEK_API_KEY，请在 .env 文件或环境变量中配置。")
        return

    print("=" * 50)
    print("  个人知识库 RAG 问答 Agent")
    print("=" * 50)

    # -- 1. 数据准备 --
    print("\n[1/4] 数据准备...")
    raw_docs, chunks = prepare_data(
        data_dir=cfg.data_dir,
        pattern=cfg.glob_pattern,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    if not chunks:
        print("请将 Markdown 笔记放入 'notes/' 文件夹后重新运行。")
        return

    # -- 2. 索引构建 --
    print("\n[2/4] 索引构建...")
    vectorstore = load_or_build_index(
        chunks=chunks,
        embedding_model=cfg.embedding_model,
        embedding_device=cfg.embedding_device,
        index_dir=cfg.index_dir,
    )

    # -- 3. 初始化检索组件 --
    print("\n[3/4] 初始化检索器...")
    bm25_index = _build_bm25(chunks)
    print("[OK] BM25 索引就绪")

    # -- 4. 初始化大模型 --
    print("\n[4/4] 初始化大模型...")
    llm = get_llm(
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
        api_key=cfg.llm_api_key,
    )
    print("[OK] DeepSeek 模型就绪")

    # -- 交互式问答 --
    print("\n" + "=" * 50)
    print("  一切就绪！输入问题开始对话，输入 quit 退出")
    print("=" * 50 + "\n")

    while True:
        try:
            question = input("你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        # 检索
        final_docs, query_type = retrieve(
            query=question,
            vectorstore=vectorstore,
            all_chunks=chunks,
            bm25_index=bm25_index,
            top_k=cfg.retrieval_top_k,
            bm25_top_k=cfg.bm25_top_k,
            final_top_k=cfg.final_top_k,
            rrf_k=cfg.rrf_k,
        )

        # 生成
        answer = generate(
            question=question,
            retrieved_docs=final_docs,
            llm=llm,
            query_type=query_type,
        )

        print(f"\n回答:\n{answer}\n")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
