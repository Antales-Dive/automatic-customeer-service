"""Agent 组装工厂（核心模块）。

每个请求动态创建 Agent，而非全局单例：
- transfer_human 需要知道当前 session_id（闭包注入）
- 不同会话可注入不同上下文

模型、工具、中间件的组合是业务核心，用户自行维护。
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
)
from circuit_breaker import get_circuit_breaker
from langchain.chat_models import init_chat_model

from config import (
    FALLBACK_MODEL,
    MAX_RETRIES,
    PII_TYPE,
    PRIMARY_MODEL,
    SUMMARY_TRIGGER_TOKENS,
    SYSTEM_PROMPT,
)
from middleware import input_guard, output_guard, tool_audit
from tools import get_weather, make_transfer_human, query_order


async def build_agent(
    *,
    session_id: str,
    mcp_tools: list | None = None,
    rag_tool=None,
) -> object:
    """构建一个绑定指定会话的 Agent。

    参数:
        session_id: 当前会话 ID（用于 transfer_human 创建工单）
        mcp_tools: 启动时加载好的 MCP 工具列表（可选）
        rag_tool: 启动时构建好的 FAQ 检索工具（可选）
    返回:
        LangChain Agent（create_agent 的结果）
    """
    primary = init_chat_model(PRIMARY_MODEL, temperature=0)
    fallback = init_chat_model(FALLBACK_MODEL, temperature=0, max_tokens=200)

    # 闭包注入 session_id：每个会话的 transfer_human 记工单到正确会话
    transfer = make_transfer_human(session_id)

    tools = [get_weather, query_order, transfer]
    if rag_tool is not None:
        tools.append(rag_tool)
    if mcp_tools:
        tools.extend(mcp_tools)

    return create_agent(
        model=primary,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            # 熔断放最外层（模块级单例，跨请求共享状态）：
            # 连续失败达阈值后直接短路，拦截对重试/降级的整个调用，
            # 避免模型服务故障时每个请求都白白重试/降级多次。
            # 注意：熔断状态必须跨请求共享，否则失败计数会丢失、熔断永不生效。
            get_circuit_breaker(),
            input_guard,
            PIIMiddleware(PII_TYPE),
            SummarizationMiddleware(
                model=fallback,
                trigger=("tokens", SUMMARY_TRIGGER_TOKENS),
            ),
            ModelRetryMiddleware(max_retries=MAX_RETRIES),
            ModelFallbackMiddleware(fallback),
            output_guard,
            tool_audit,
        ],
    )
