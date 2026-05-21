"""
个人知识库 RAG 问答 Agent — 配置管理
所有可调参数集中在这里，修改配置不用翻代码。
"""
import os
from dotenv import load_dotenv

load_dotenv()

# HuggingFace 镜像（国内加速），留空则不使用镜像
HF_ENDPOINT = "https://hf-mirror.com"
if HF_ENDPOINT:
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT


class RAGConfig:
    """RAG 系统全局配置"""

    # ── 路径 ──
    data_dir: str = "notes"                     # 笔记文件夹
    index_dir: str = "vector_index"             # FAISS 索引保存路径
    glob_pattern: str = "**/*.md"               # 要加载的文件类型

    # ── 分块 ──
    chunk_size: int = 500                       # 子块大小（字符数）
    chunk_overlap: int = 50                     # 子块重叠

    # ── 嵌入模型 ──
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"

    # ── 大模型 ──
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 1.0
    llm_max_tokens: int = 2048
    llm_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")

    # ── 检索 ──
    retrieval_top_k: int = 5                    # 向量检索返回数
    bm25_top_k: int = 5                         # BM25 返回数
    final_top_k: int = 3                        # RRF 重排后最终数量
    rrf_k: int = 60                             # RRF 平滑参数

    # ── 提示词模板 ──
    system_prompt: str = (
        "你是我的个人知识库助手。请根据下面提供的笔记内容回答问题。\n"
        "规则：\n"
        "1. 回答必须基于提供的上下文，不要使用外部知识。\n"
        "2. 如果上下文不足以回答问题，请直接告知：'抱歉，我的笔记中没有找到相关信息。'\n"
        "3. 回答要简洁、结构化，尽量列出要点。"
    )
