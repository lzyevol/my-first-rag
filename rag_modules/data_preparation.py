"""
数据准备模块 — 加载 Markdown 笔记、按标题分块、建立父子文档关系。
策略：用 Markdown 标题分块作为子块（精确检索），整个文档作为父块（完整上下文）。
"""
import os
import glob
from typing import List, Tuple

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def load_markdown_files(data_dir: str, pattern: str = "**/*.md") -> List[Document]:
    """
    扫描目录下所有 Markdown 文件并加载。
    返回原始文档列表，每个文件一个 Document。
    """
    all_docs = []
    search_path = os.path.join(data_dir, pattern)
    md_files = glob.glob(search_path, recursive=True)

    if not md_files:
        print(f"[WARN] 在 '{data_dir}' 中没有找到任何 .md 文件，请先放入笔记。")
        return all_docs

    print(f"[INFO] 找到 {len(md_files)} 个 Markdown 文件")
    for fpath in md_files:
        loader = TextLoader(fpath, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = fpath
            doc.metadata["filename"] = os.path.basename(fpath)
        all_docs.extend(docs)
    return all_docs


def split_by_headers(docs: List[Document]) -> List[Document]:
    """
    按 Markdown 标题层级分块（子块），并保留父文档信息。
    标题层级：H1(#) / H2(##) / H3(###)
    """
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,   # 保留标题文字在正文中
    )

    child_chunks = []
    for doc in docs:
        # 为每个文档按标题切分
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            # 子块继承父文档元数据 + 自己的标题层级
            chunk.metadata["source"] = doc.metadata.get("source", "")
            chunk.metadata["filename"] = doc.metadata.get("filename", "")
            # 标记这是一个子块，存父文档原文引用
            chunk.metadata["parent_content"] = doc.page_content
            child_chunks.append(chunk)

    print(f"[INFO] 按标题分块后得到 {len(child_chunks)} 个子块")
    return child_chunks


def split_fallback(chunks: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    对于过长的子块（超过 chunk_size），递归字符分割兜底。
    """
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    final_chunks = []
    for chunk in chunks:
        if len(chunk.page_content) > chunk_size:
            sub_chunks = fallback_splitter.split_documents([chunk])
            for sub in sub_chunks:
                # 子块继承原块的全部元数据
                sub.metadata = {**chunk.metadata}
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    print(f"[INFO] 最终得到 {len(final_chunks)} 个文本块（含兜底分割）")
    return final_chunks


def prepare_data(
    data_dir: str = "notes",
    pattern: str = "**/*.md",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Tuple[List[Document], List[Document]]:
    """
    数据准备主流程。
    返回: (原始文档列表, 最终文本块列表)
    """
    # 1. 加载所有 Markdown 文件
    raw_docs = load_markdown_files(data_dir, pattern)
    if not raw_docs:
        return [], []

    # 2. 按 Markdown 标题分块
    header_chunks = split_by_headers(raw_docs)

    # 3. 过长块兜底分割
    final_chunks = split_fallback(header_chunks, chunk_size, chunk_overlap)

    return raw_docs, final_chunks
