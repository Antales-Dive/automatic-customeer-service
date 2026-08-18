"""Pydantic 请求/响应模型。

纯数据结构定义，无业务逻辑。由 AI 生成，用户只需理解每个字段的含义。
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /api/chat 的请求体：用户发送的一条消息。"""

    content: str = Field(..., min_length=1, max_length=5000, description="用户消息内容")


class ChatResponse(BaseModel):
    """POST /api/chat 的响应体：AI 的回复。"""

    role: str = "assistant"
    content: str
    session_id: str


class SessionInfo(BaseModel):
    """会话列表项。"""

    session_id: str
    title: str
    updated_at: str
    message_count: int = 0


class SessionDetail(BaseModel):
    """会话详情：标题 + 全部消息。"""

    session_id: str
    title: str
    messages: list[dict]


class TicketCreate(BaseModel):
    """POST /api/tickets 的请求体：创建工单。"""

    session_id: str
    user_message: str = Field(..., min_length=1, description="用户转人工时的问题描述")
    priority: str = "medium"


class TicketInfo(BaseModel):
    """工单信息。"""

    id: str
    session_id: str
    user_message: str
    status: str
    priority: str
    created_at: str
