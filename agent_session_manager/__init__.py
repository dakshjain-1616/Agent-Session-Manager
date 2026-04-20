"""
Stateful Agent Session Manager

A lightweight Python library that persists agent state across sessions
using SQLite and a local ChromaDB vector store.

Example:
    >>> from agent_session_manager import SessionManager
    >>> 
    >>> manager = SessionManager()
    >>> state = manager.create_session(
    ...     session_id="session-001",
    ...     agent_id="agent-1",
    ...     initial_goal="Build a chatbot"
    ... )
    >>> manager.add_step(session_id="session-001", action="Initialize model")
    >>> 
    >>> # Later, restore the session
    >>> restored = manager.load_session("session-001")
    >>> print(f"Goal: {restored.current_goal}")
    >>> print(f"Steps: {len(restored.completed_steps)}")
"""

from .persistence import SQLitePersistence
from .memory import ChromaDBMemory
from .manager import (
    SessionManager,
    AgentState,
    SessionManagerError,
    SessionNotFoundError,
    BudgetExceededError
)

__version__ = "0.1.0"
__all__ = [
    "SessionManager",
    "AgentState", 
    "SQLitePersistence",
    "ChromaDBMemory",
    "SessionManagerError",
    "SessionNotFoundError",
    "BudgetExceededError"
]
