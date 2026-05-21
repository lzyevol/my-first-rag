"""
检索模块 — 混合检索（向量 + BM25）+ RRF 重排序 + 简单查询路由。
"""
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


# ── BM25 关键词检索 ──

def _build_bm25(chunks: List[Document]) -> BM25Okapi:
    """用子块文本构建 BM25 索引。"""
    texts = [chunk.page_content for chunk in chunks]
    return BM25Okapi([t.split() for t in texts])


def bm25_search(query: str, chunks: List[Document], bm25_index: BM25Okapi, top_k: int = 5) -> List[Document]:
    """BM25 关键词检索，返回 top_k 个文档。"""
    tokenized_query = query.split()
    scores = bm25_index.get_scores(tokenized_query)
    # 取分数最高的 top_k 个
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results = []
    for idx in ranked_indices[:top_k]:
        if scores[idx] > 0:
            chunk = chunks[idx]
            chunk.metadata["bm25_score"] = float(scores[idx])
            results.append(chunk)
    return results


# ── 向量检索 ──

def vector_search(query: str, vectorstore: FAISS, top_k: int = 5) -> List[Document]:
    """FAISS 向量相似度检索。"""
    return vectorstore.similarity_search(query, k=top_k)


# ── RRF 重排序 ──

def rrf_fusion(
    bm25_results: List[Document],
    vector_results: List[Document],
    k: int = 60,
    final_top_k: int = 3,
) -> List[Document]:
    """
    Reciprocal Rank Fusion：融合两个检索结果。
    不看分数只看排名，两边排名都靠前的文档胜出。
    """
    rrf_scores = {}

    for rank, doc in enumerate(bm25_results):
        doc_id = doc.page_content[:100]  # 用内容前 100 字符做去重 key
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rank + k)

    for rank, doc in enumerate(vector_results):
        doc_id = doc.page_content[:100]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (rank + k)

    # 按 RRF 分数降序排列
    seen = set()
    merged: List[Tuple[Document, float]] = []

    for doc in bm25_results + vector_results:
        doc_id = doc.page_content[:100]
        if doc_id not in seen:
            seen.add(doc_id)
            merged.append((doc, rrf_scores.get(doc_id, 0.0)))

    merged.sort(key=lambda x: x[1], reverse=True)

    # 返回最终 top_k 个文档，如果内容为空则尝试用父文档替换
    final_docs = []
    for doc, _ in merged[:final_top_k]:
        # 如果子块太短，用父文档原文替代（父子文档策略）
        if len(doc.page_content.strip()) < 50 and doc.metadata.get("parent_content"):
            parent_doc = Document(
                page_content=doc.metadata["parent_content"],
                metadata={**doc.metadata},
            )
            final_docs.append(parent_doc)
        else:
            final_docs.append(doc)

    return final_docs


# ── 简单查询路由 ──

def route_query(query: str) -> str:
    """
    基于关键词判断查询类型：
    - list: 列举/推荐类
    - detail: 怎么做/步骤
    - general: 其他
    """
    list_keywords = ["有哪些", "列举", "推荐", "所有", "分类", "几种"]
    detail_keywords = ["怎么做", "步骤", "流程", "如何", "怎样", "方法", "区别", "对比"]

    for kw in list_keywords:
        if kw in query:
            return "list"
    for kw in detail_keywords:
        if kw in query:
            return "detail"
    return "general"


# ── 主检索入口 ──

def retrieve(
    query: str,
    vectorstore: FAISS,
    all_chunks: List[Document],
    bm25_index: BM25Okapi,
    top_k: int = 5,
    bm25_top_k: int = 5,
    final_top_k: int = 3,
    rrf_k: int = 60,
) -> Tuple[List[Document], str]:
    """
    检索主流程：路由判断 → 混合检索 → RRF 融合。
    返回: (最终文档列表, 查询类型)
    """
    query_type = route_query(query)

    # BM25 关键词检索
    bm25_results = bm25_search(query, all_chunks, bm25_index, top_k=bm25_top_k)

    # FAISS 向量检索
    vector_results = vector_search(query, vectorstore, top_k=top_k)

    # RRF 融合
    final_docs = rrf_fusion(bm25_results, vector_results, k=rrf_k, final_top_k=final_top_k)

    # 如果检索结果为空，尝试用 BM25 结果兜底
    if not final_docs:
        final_docs = bm25_results[:final_top_k]

    return final_docs, query_type
