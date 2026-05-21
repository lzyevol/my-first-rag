# 个人知识库 RAG 问答 Agent

基于 All-in-RAG 学习后独立实现的轻量级 RAG 问答系统，支持将个人 Markdown 笔记构建为可检索的知识库，通过 DeepSeek 大模型生成基于笔记内容的回答。

## 技术栈

- **文档处理**：LangChain（MarkdownHeaderTextSplitter 标题感知切分 + RecursiveCharacterTextSplitter 兜底）
- **向量化**：BAAI/bge-small-zh-v1.5（HuggingFace Embedding）
- **向量数据库**：FAISS（本地持久化）
- **混合检索**：FAISS 向量检索 + BM25 关键词检索 + RRF 融合排序
- **大模型**：DeepSeek Chat

## 项目结构

```
my-rag-agent/
├── main.py                 # 主入口，串联四模块流程
├── config.py               # 全局配置（模型、分块、检索参数）
├── requirements.txt        # 依赖清单
├── .env                    # API 密钥（不提交）
├── notes/                  # 你的 Markdown 笔记
├── rag_modules/            # 核心模块
│   ├── data_preparation.py # 文档加载 + 标题切分
│   ├── index_construction.py # 向量化 + FAISS 索引
│   ├── retrieval.py        # 混合检索 + RRF 重排
│   └── generation.py       # LLM 生成回答
└── vector_index/           # 自动生成的索引文件
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（编辑 .env 文件）
DEEPSEEK_API_KEY=你的密钥

# 3. 放入笔记到 notes/ 文件夹（Markdown 格式，建议有标题层级）

# 4. 运行
python main.py
```

## 特性

- Markdown 标题层级感知的文本切分，超长块递归兜底
- FAISS + BM25 混合检索，RRF 融合两路结果
- 父子文档策略：小块精确检索，大块完整生成
- 防幻觉三层防线：低 temperature + 分层 prompt + 内容量检查

## 学习背景

本项目是 [All-in-RAG](https://github.com/datawhalechina/all-in-rag) 课程的练手实践项目。
