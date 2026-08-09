"""集中配置：所有环境适配相关的常量都放这里。

设计原则：可能随部署环境变化的量（路径、API 配置、业务常量）统一在此定义，
其他模块从这里 import，避免散落硬编码。换环境只需改这一个文件。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ── 环境变量加载 ──
# find_dotenv() 从当前目录向上查找 .env，保证从任意子目录启动都能加载
load_dotenv()

# ── 路径 ──
# 项目根 = 07_customer_service 的上级（.env、mcp_workspace 都在这）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 本目录
APP_DIR = Path(__file__).resolve().parent

MCP_WORKSPACE = str(PROJECT_ROOT / "mcp_workspace")   # MCP 文件系统允许访问的目录
CHROMA_PERSIST_DIR = str(APP_DIR / "chroma_store")     # 向量库持久化目录
FAQ_DOCS_DIR = str(APP_DIR / "faq_docs")               # FAQ 知识库文档目录
DB_PATH = str(APP_DIR / "customer_service.db")          # SQLite 数据库文件
STATIC_DIR = str(APP_DIR / "static")                   # 前端静态文件目录

# ── 模型 ──
PRIMARY_MODEL = "deepseek:deepseek-v4-flash"   # 主模型
FALLBACK_MODEL = "deepseek:deepseek-v4-pro"    # 降级/摘要用模型

# ── 中间件参数（业务决策，用户自定义）──
BANNED_WORDS = ["炸药", "诈骗", "毒品", "违禁词1", "违禁词2"]  # 输入拦截词表
PII_TYPE = "email"         # PII 脱敏类型
MAX_RETRIES = 3            # 模型重试次数
SUMMARY_TRIGGER_TOKENS = 3000  # 历史超过多少 token 触发摘要

# ── 熔断（Circuit Breaker）──
# 连续失败达到阈值后，一段时间内直接短路，不再调用模型，服务快速失败。
CIRCUIT_FAILURE_THRESHOLD = 3    # 连续失败多少次后熔断打开
CIRCUIT_COOL_DOWN_SECONDS = 30.0 # 熔断打开后多久进入半开试探
CIRCUIT_OPEN_MESSAGE = "😥 服务当前繁忙，请稍后再试或转人工客服。"

# ── RAG ──

# ── RAG ──
RAG_K = 3                          # 检索返回条数
CHUNK_SIZE = 200                   # 文本切分块大小
CHUNK_OVERLAP = 20                 # 切分重叠
COLLECTION_NAME = "customer_faq"   # 向量库集合名

# 检索置信度阈值（方案一：防幻觉）
# 含义：检索分数（Chroma 中为 1 - 欧氏距离）低于该值视为「知识库没有答案」，
#       工具会返回兜底提示并建议转人工，避免 Agent 拿着不相关上下文硬编答案。
# 标定方法：运行 scripts/calibrate_threshold.py 看真实问题/无关问题的分数分布，
#          一般选「能正确回答问题的分数」与「明显无关的分数」之间的值，再留点余量。
RAG_SCORE_THRESHOLD = 0.5

# ── 系统提示词（客服人设，业务决策）──
SYSTEM_PROMPT = (
    "你是智能客服助手，帮用户解决常见问题。\n"
    "- 回答知识库问题前，先调用 search_faq 检索；必须基于检索结果回答，不要编造。\n"
    "- search_faq 返回结果为空或开头是\"知识库没有检索到相关答案\"时，"
    "说明知识库里没有该问题的答案，不要硬编，应告知用户并调用 transfer_human 转人工。\n"
    "- 以下情况不要调用 search_faq：用户只是想闲聊、查询天气、查询订单状态、读写文件，"
    "或明确要求转人工/投诉。\n"
    "- 用户明确要求转人工、投诉或你无法解决问题时，调用 transfer_human 工具。\n"
    "- 回答简洁友好，可引用检索到的来源文件名（如「根据《faq.txt》」）以增强可信度。"
)
