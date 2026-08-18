"""SQLite 数据访问层（aiosqlite 异步）。

用户设计函数签名（业务需要哪些查询），SQL 实现为模板。
每个函数使用独立连接，简单可靠；MVP 规模下性能足够。
"""

import sqlite3
import uuid
from datetime import datetime, timezone

import aiosqlite

# ── 建表 ──

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT DEFAULT '新会话',
    status      TEXT DEFAULT 'active',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(id),
    user_message  TEXT NOT NULL,
    priority      TEXT DEFAULT 'medium',
    status        TEXT DEFAULT 'pending',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
"""


def _now() -> str:
    """返回 ISO 8601 UTC 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


async def init_db(db_path: str) -> None:
    """应用启动时调用：创建表。"""
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(_CREATE_TABLES)
        await db.commit()


# ── 会话 ──

async def create_session(db_path: str) -> dict:
    """创建新会话，返回会话信息。"""
    session_id = str(uuid.uuid4())
    now = _now()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO sessions (id, title, status, created_at, updated_at) "
            "VALUES (?, '新会话', 'active', ?, ?)",
            (session_id, now, now),
        )
        await db.commit()
    return {"id": session_id, "title": "新会话", "created_at": now}


async def list_sessions(db_path: str) -> list[dict]:
    """列出所有会话（含消息数），按更新时间倒序。"""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cur = await db.execute(
            "SELECT s.id, s.title, s.updated_at, COUNT(m.id) AS message_count "
            "FROM sessions s LEFT JOIN messages m ON s.id = m.session_id "
            "GROUP BY s.id ORDER BY s.updated_at DESC"
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


async def get_session(db_path: str, session_id: str) -> dict | None:
    """获取单个会话信息；不存在返回 None。"""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cur = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = await cur.fetchone()
        return dict(row) if row else None


async def update_session_title(db_path: str, session_id: str, title: str) -> None:
    """更新会话标题（首条消息时自动生成）。"""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), session_id),
        )
        await db.commit()


async def touch_session(db_path: str, session_id: str) -> None:
    """仅更新 updated_at（会话活跃时间）。"""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (_now(), session_id),
        )
        await db.commit()


# ── 消息 ──

async def add_message(db_path: str, session_id: str, role: str, content: str) -> None:
    """写入一条消息。"""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )
        await db.commit()


async def get_messages(db_path: str, session_id: str) -> list[dict]:
    """获取会话全部消息，按时间正序（对话顺序）。"""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cur = await db.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE session_id = ? ORDER BY created_at ASC, id ASC",
            (session_id,),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


# ── 工单 ──

async def create_ticket(
    db_path: str, session_id: str, user_message: str, priority: str = "medium"
) -> dict:
    """创建工单，返回工单信息。"""
    ticket_id = str(uuid.uuid4())
    now = _now()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO tickets (id, session_id, user_message, priority, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 'pending', ?, ?)",
            (ticket_id, session_id, user_message, priority, now, now),
        )
        await db.commit()
    return {
        "id": ticket_id,
        "session_id": session_id,
        "user_message": user_message,
        "priority": priority,
        "status": "pending",
        "created_at": now,
    }


async def list_tickets(db_path: str) -> list[dict]:
    """列出所有工单，按创建时间倒序。"""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        cur = await db.execute(
            "SELECT id, session_id, user_message, status, priority, created_at "
            "FROM tickets ORDER BY created_at DESC"
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]
