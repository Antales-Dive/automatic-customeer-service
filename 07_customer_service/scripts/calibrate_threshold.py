"""阈值标定脚本（方案一配套）。

用法：
    1) 确保你的 venv 已激活、Ollama 在运行（start.bat 第 1/2 步）。
    2) python scripts/calibrate_threshold.py

它会用「知识库相关的问题」和「明显无关的问题」各跑几条，
打印每条检索结果的分数，方便你判断 RAG_SCORE_THRESHOLD 该取多少。
"""

import os
import sys
from pathlib import Path

# 把项目根目录加入 sys.path，保证能 import 到 config / rag
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    RAG_K,
    RAG_SCORE_THRESHOLD,
)
from rag import build_embeddings, load_faq_documents, build_vectorstore

# 知识库相关的真实问题（应该能检索到、分数高）
relevant_queries = [
    "退换货怎么办理？",
    "支付不成功怎么办？",
    "忘记密码如何找回？",
    "订单发货要多久？",
]

# 明显无关的问题（知识库里没有，分数应该低）
irrelevant_queries = [
    "今天天气怎么样？",
    "你会写诗吗？",
    "推荐一部好看的电影。",
    "2+2等于几？",
]


def main() -> None:
    print("=" * 60)
    print("RAG 阈值标定工具")
    print("=" * 60)

    # 复用 / 构建向量库（和启动时同一套逻辑）
    docs = load_faq_documents()
    vectorstore = build_vectorstore(docs)
    print(f"向量库文档数: {len(docs)}  当前阈值: {RAG_SCORE_THRESHOLD}\n")
    print("说明：分数 = 1 - 欧氏距离，分数越高越相似。\n"
          "agent 的 search_faq 会用 score_threshold 过滤掉低于阈值的片段。\n")

    # 用 similarity_search_with_score 拿到每条的原始分数（不做阈值过滤），
    # 这样能直接观察「相关问题」和「无关问题」的分数分布。
    def _print_scores(queries: list[str], label: str) -> None:
        print(f"【{label}】")
        for q in queries:
            results = vectorstore.similarity_search_with_score(q, k=RAG_K)
            print(f"\nQ: {q}")
            if not results:
                print("    (无结果)")
                continue
            for doc, score in results:
                sim = round(1 - score, 4)  # 转换回阈值使用的分数
                print(f"    score={sim}  {doc.page_content[:36]}...")

    _print_scores(relevant_queries, "知识库相关问题——分数应当较高")
    print("\n" + "=" * 60)
    _print_scores(irrelevant_queries, "无关问题——分数应当较低")

    print("\n" + "=" * 60)
    print("参考取值：在「相关问题的最低分」和「无关问题的最高分」之间选一个值，")
    print("再留一点余量。改 config.py 里的 RAG_SCORE_THRESHOLD 即可。")
    print("=" * 60)


if __name__ == "__main__":
    main()
