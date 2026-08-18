"""RAG 知识库：FAQ 文档加载 + Chroma 向量库 + 检索工具。

管道（切分/embedding/向量库）参考 04_RAG/main.py 的已验证模式。
从 config.py 读取路径和配置，使本模块可复用、可替换。
"""

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    FAQ_DOCS_DIR,
    RAG_K,
    RAG_SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ── 1. FAQ 文档加载 ──

def load_faq_documents() -> list[Document]:
    """从 faq_docs/ 目录读取所有 .txt 文件，切分为问答文档。"""
    docs: list[Document] = []
    from pathlib import Path
    faq_dir = Path(FAQ_DOCS_DIR)
    for txt_path in sorted(faq_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8")
        # 每条 "Q:..." 切成一则文档，保留题目作为元数据
        for block in text.split("Q:"):
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()
            question = lines[0].strip()
            # 去掉答案开头的 "A:" 前缀（如果存在）
            answer_lines = lines[1:]
            if answer_lines and answer_lines[0].strip().startswith("A:"):
                answer_lines[0] = answer_lines[0].strip()[2:].strip()
            answer = "\n".join(answer_lines).strip()
            docs.append(
                Document(
                    page_content=f"Q: {question}\nA: {answer}",
                    metadata={"source": txt_path.name, "question": question},
                )
            )
    if not docs:
        logger.warning("faq_docs/ 下没有 FAQ 文档，知识库为空。")
    return docs


# ── 2. 向量库（带持久化检测，避免重复 embedding）──

def build_embeddings():
    """Ollama embedding（从环境变量读取 base_url / model）。"""
    import os
    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL"),
        base_url=os.getenv("OLLAMA_BASE_URL"),
    )


def build_vectorstore(docs: list[Document]) -> Chroma:
    """构建或复用已持久化的向量库。"""
    embeddings = build_embeddings()

    # 仅当已有持久化且目标集合非空时才复用，否则重建
    from pathlib import Path
    db_file = Path(CHROMA_PERSIST_DIR) / "chroma.sqlite3"
    if db_file.exists():
        existing = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        if existing._collection.count() > 0:
            logger.info("复用已有向量库：%s", CHROMA_PERSIST_DIR)
            return existing
        logger.info("集合 %s 为空，重建向量库", COLLECTION_NAME)

    logger.info("构建新向量库（%d 篇文档）", len(docs))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )


# ── 3. 检索工具 ──

def build_faq_search_tool(vectorstore: Chroma):
    """把向量库包装成 Agent 可用的检索工具。

    方案一（防幻觉）：使用「相似度分数阈值」检索。
    - 检索分数低于 RAG_SCORE_THRESHOLD 的结果会被 Chroma 直接过滤掉；
    - 过滤后若为空，说明知识库里没有这个问题，返回明确提示，
      引导 Agent 不要硬编答案，而是走转人工流程。
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": RAG_K, "score_threshold": RAG_SCORE_THRESHOLD},
    )
    no_answer_prefix = "知识库没有检索到相关答案"

    def _search(query: str) -> str:
        results = retriever.invoke(query)
        if not results:
            return (
                f"{no_answer_prefix}，建议告知用户当前问题不在常见问题范围内，"
                "并主动引导用户转人工客服处理。"
            )
        # 拼接检索到的 FAQ 条目，每段保留来源文件名，方便 Agent 回答时引用
        blocks = []
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "未知来源")
            blocks.append(f"[来源：{source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(blocks)

    return StructuredTool.from_function(
        func=_search,
        name="search_faq",
        description=(
            "搜索客服 FAQ 知识库，获取常见问题的标准答案。"
            "当用户询问以下主题时使用：账号与登录、退换货、支付与退款、"
            "物流与配送、订单管理、客服工作时间、企业合作。"
            "注意：如果返回结果为空或开头是\"知识库没有检索到相关答案\"，"
            "说明知识库中没有该问题的答案，不要编造，应告知用户并转人工。"
        ),
    )
