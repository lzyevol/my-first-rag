"""
索引构建模块 — 使用 BGE 嵌入模型将文本块转向量，存入 FAISS 索引。
支持增量构建：如果索引文件已存在则直接加载，否则新建。
"""
import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def get_embeddings(model_name: str, device: str = "cpu") -> HuggingFaceEmbeddings:
    """初始化 BGE 嵌入模型（单例，避免重复加载）。"""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    index_dir: str,
) -> FAISS:
    """从文本块列表构建 FAISS 索引并保存。"""
    print(f"[INFO] 正在构建 FAISS 索引，共 {len(chunks)} 个文本块...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_dir)
    print(f"[OK] 索引已保存到 '{index_dir}'")
    return vectorstore


def load_or_build_index(
    chunks: List[Document],
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    embedding_device: str = "cpu",
    index_dir: str = "vector_index",
) -> FAISS:
    """
    索引管理主入口：
    - 如果 index_dir 下有已有索引 → 直接加载
    - 如果没有 → 从 chunks 新建
    返回 FAISS 向量库对象。
    """
    embeddings = get_embeddings(embedding_model, embedding_device)

    index_file = os.path.join(index_dir, "index.faiss")
    if os.path.exists(index_file):
        print(f"[INFO] 发现已有索引，直接加载 '{index_dir}' ...")
        vectorstore = FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"[OK] 索引加载完成")
        return vectorstore

    if not chunks:
        raise ValueError("没有文本块可用，且索引不存在。请先放入笔记文件再运行。")

    return build_index(chunks, embeddings, index_dir)
