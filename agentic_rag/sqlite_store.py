import sqlite3
from typing import List, Dict

DB_PATH = "agentic_rag.db"


def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT,
        role TEXT,
        content TEXT,
        tool_calls TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT,
        question TEXT,
        answer TEXT,
        reward REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


# -------------------------
# Conversation persistence
# -------------------------

def save_message(
    conversation_id: str,
    role: str,
    content: str,
    tool_calls: str | None = None,
):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO messages (conversation_id, role, content, tool_calls)
        VALUES (?, ?, ?, ?)
        """,
        (conversation_id, role, content, tool_calls),
    )
    conn.commit()
    conn.close()


def load_conversation(conversation_id: str) -> List[Dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT role, content, tool_calls
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
        """,
        (conversation_id,),
    )

    rows = cur.fetchall()
    conn.close()

    return [
        {
            "role": r[0],
            "content": r[1],
            "tool_calls": r[2],
        }
        for r in rows
    ]


def save_feedback(conversation_id, question, answer, reward):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO feedback (conversation_id, question, answer, reward)
        VALUES (?, ?, ?, ?)
        """,
        (conversation_id, question, answer, reward),
    )
    conn.commit()
    conn.close()
