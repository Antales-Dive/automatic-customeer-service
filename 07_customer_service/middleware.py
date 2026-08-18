"""自定义中间件：输入拦截、输出质检、工具审计。

直接从 06_/main.py 迁移（已验证可运行）。
注意两个已知坑：
- 钩子返回的键是 "messages"（列表），不是 "message"（否则被静默忽略）
- after_model 的 output_guard 必须跳过 tool_calls 消息（content 为空会误判"过短"）
"""

from langchain.agents.middleware import before_model, after_model, wrap_tool_call
from langchain_core.messages import AIMessage

from config import BANNED_WORDS


@before_model(can_jump_to=["end"])
def input_guard(state, runtime):
    """输入拦截：命中违禁词直接终止，不再调用模型。"""
    last_user = [m for m in state["messages"] if m.type == "human"][-1]
    if any(w in last_user.content for w in BANNED_WORDS):
        return {"jump_to": "end", "messages": [AIMessage(content="🚫 输入违规，已拦截。")]}
    return None


@after_model(can_jump_to=["end", "model"])
def output_guard(state, runtime):
    """输出质检：内容违规则终止；回答过短则让模型重写一次（防死循环）。

    after_model 在模型每次返回后触发，包括带 tool_calls 的消息
    （content 为空），必须跳过，否则打断工具调用流程。
    """
    last = state["messages"][-1]
    if last.type != "ai":
        return None
    if last.tool_calls:
        return None
    if any(w in last.content for w in ["敏感内容", "违规"]):
        return {"jump_to": "end", "messages": [AIMessage(content="🚫 输出违规。")]}
    if not state.get("_reviewed") and len(last.content) < 10:
        state["_reviewed"] = True
        return {"jump_to": "model"}
    return None


@wrap_tool_call
async def tool_audit(request, handler):
    """工具审计：记录工具名和参数，异常时返回兜底信息。

    必须是 async 版本：Agent 用 ainvoke（异步）调用。
    """
    print(f"[AUDIT] 工具={request.tool.name} 参数={request.tool_call['args']}")
    try:
        return await handler(request)
    except Exception as e:
        return f"工具执行失败：{e}"
