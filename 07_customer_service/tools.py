"""业务工具。

设计要点：
- get_weather / query_order：模拟数据，接口设计为后续可替换真实 API/数据库。
- make_transfer_human(session_id)：闭包工厂，把会话 ID 注入工具，
  使 transfer_human 能针对「当前会话」创建工单并写入数据库。
"""

import asyncio

from langchain_core.tools import tool

from database import create_ticket
from config import DB_PATH


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    data = {"北京": "晴，25°C", "上海": "多云，22°C", "广州": "雷阵雨，28°C"}
    return data.get(city, f"暂无{city}的天气信息")


@tool
def query_order(order_id: str) -> str:
    """根据订单号查询订单状态。"""
    statuses = {"A1001": "已发货", "A1002": "待付款", "A1003": "已签收"}
    return f"订单 {order_id} 状态：{statuses.get(order_id, '未找到该订单')}"


def make_transfer_human(session_id: str):
    """创建绑定了指定会话的 transfer_human 工具。

    使用闭包把 session_id 注入工具内部，这样 Agent 在任意会话中调用
    transfer_human 时，工单都会记到正确的会话下。
    """

    @tool
    def transfer_human(message: str) -> str:
        """将问题转接给人工客服处理。当用户明确要求转人工、投诉或问题无法解决时调用。

        参数:
            message: 用户转人工时的问题描述或诉求。
        返回:
            工单编号和提示信息。
        """
        # 工具是同步函数，但 create_ticket 是 async：用 asyncio.run 桥接。
        # （此处无其他运行中的事件循环，安全；若将来有则改用 run_coroutine_threadsafe）
        ticket = asyncio.run(create_ticket(DB_PATH, session_id, message))
        return f"已将您的问题转接人工客服，工单号：{ticket['id']}，我们会尽快处理。"

    return transfer_human
