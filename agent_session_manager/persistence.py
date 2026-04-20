"""
SQLite Persistence Layer for Agent Session Manager.

Provides CRUD operations for:
- Session state storage
- Step/action tracking
- Key-value storage for arbitrary data
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from contextlib import contextmanager
import threading


class SQLitePersistence:
    """
    Thread-safe SQLite persistence layer for agent sessions.
    
    Manages:
    - sessions: Core session metadata and state
    - steps: Individual steps/actions within a session
    - kv_store: Key-value storage for arbitrary session data
    """
    
    def __init__(self, db_path: str = "agent_sessions.db"):
        """
        Initialize SQLite persistence.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def _init_db(self):
        """Initialize database schema."""
        with self._transaction() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    context_data TEXT,
                    metadata TEXT
                )
            """)
            
            # Steps table for tracking actions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS steps (
                    step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    result TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            
            # Key-value store for arbitrary data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, key),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_steps_session ON steps(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_steps_number ON steps(session_id, step_number)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_kv_session ON kv_store(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_id)")
    
    # Session Operations
    def create_session(self, session_id: str, agent_id: str, 
                       context_data: Optional[Dict] = None,
                       metadata: Optional[Dict] = None) -> bool:
        """
        Create a new session.
        
        Args:
            session_id: Unique session identifier
            agent_id: Agent identifier
            context_data: Initial context data (JSON-serializable)
            metadata: Session metadata (JSON-serializable)
            
        Returns:
            True if created successfully, False if session_id already exists
        """
        try:
            with self._transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT INTO sessions (session_id, agent_id, context_data, metadata)
                       VALUES (?, ?, ?, ?)""",
                    (session_id, agent_id,
                     json.dumps(context_data) if context_data else None,
                     json.dumps(metadata) if metadata else None)
                )
                return True
        except sqlite3.IntegrityError:
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dict or None if not found
        """
        with self._transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return self._row_to_dict(row)
    
    def update_session(self, session_id: str, 
                       context_data: Optional[Dict] = None,
                       status: Optional[str] = None,
                       metadata: Optional[Dict] = None) -> bool:
        """
        Update session data.
        
        Args:
            session_id: Session identifier
            context_data: Updated context data
            status: Updated status
            metadata: Updated metadata (merged with existing)
            
        Returns:
            True if updated, False if session not found
        """
        with self._transaction() as conn:
            cursor = conn.cursor()
            
            # Get existing session
            cursor.execute("SELECT context_data, metadata FROM sessions WHERE session_id = ?", 
                        (session_id,))
            row = cursor.fetchone()
            if row is None:
                return False
            
            # Build update fields
            updates = ["updated_at = CURRENT_TIMESTAMP"]
            params = []
            
            if context_data is not None:
                updates.append("context_data = ?")
                params.append(json.dumps(context_data))
            
            if status is not None:
                updates.append("status = ?")
                params.append(status)
            
            if metadata is not None:
                # Merge with existing metadata
                existing_metadata = json.loads(row['metadata']) if row['metadata'] else {}
                existing_metadata.update(metadata)
                updates.append("metadata = ?")
                params.append(json.dumps(existing_metadata))
            
            params.append(session_id)
            
            cursor.execute(
                f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?",
                params
            )
            return cursor.rowcount > 0
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session and all associated data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        with self._transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            return cursor.rowcount > 0
    
    def list_sessions(self, agent_id: Optional[str] = None, 
                      status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List sessions with optional filtering.
        
        Args:
            agent_id: Filter by agent ID
            status: Filter by status
            
        Returns:
            List of session data dicts
        """
        with self._transaction() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM sessions WHERE 1=1"
            params = []
            
            if agent_id:
                query += " AND agent_id = ?"
                params.append(agent_id)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY updated_at DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_dict(row) for row in rows]
    
    # Step Operations
    def add_step(self, session_id: str, action: str, 
                 result: Optional[str] = None,
                 metadata: Optional[Dict] = None) -> int:
        """
        Add a step to a session.
        
        Args:
            session_id: Session identifier
            action: Action description
            result: Action result
            metadata: Step metadata
            
        Returns:
            Step number assigned
        """
        with self._transaction() as conn:
            cursor = conn.cursor()
            
            # Get next step number
            cursor.execute(
                "SELECT COALESCE(MAX(step_number), 0) + 1 FROM steps WHERE session_id = ?",
                (session_id,)
            )
            step_number = cursor.fetchone()[0]
            
            cursor.execute(
                """INSERT INTO steps (session_id, step_number, action, result, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, step_number, action, result,
                 json.dumps(metadata) if metadata else None)
            )
            
            # Update session timestamp
            cursor.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                (session_id,)
            )
            
            return step_number
    
    def get_steps(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get steps for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of steps to return (most recent)
            
        Returns:
            List of step data dicts ordered by step_number
        """
        with self._transaction() as conn:
            cursor = conn.cursor()
            
            query = """SELECT * FROM steps WHERE session_id = ? 
                       ORDER BY step_number ASC"""
            params = [session_id]
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_dict(row) for row in rows]
    
    def get_step_count(self, session_id: str) -> int:
        """Get number of steps in a session."""
        with self._transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM steps WHERE session_id = ?", (session_id,))
            return cursor.fetchone()[0]
    
    # KV Store Operations
    def set_kv(self, session_id: str, key: str, value: Any) -> bool:
        """
        Set key-value pair for a session.
        
        Args:
            session_id: Session identifier
            key: Key name
            value: JSON-serializable value
            
        Returns:
            True if set successfully
        """
        with self._transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO kv_store (session_id, key, value) VALUES (?, ?, ?)
                   ON CONFLICT(session_id, key) DO UPDATE SET 
                   value = excluded.value, updated_at = CURRENT_TIMESTAMP""",
                (session_id, key, json.dumps(value))
            )
            return True
    
    def get_kv(self, session_id: str, key: str, default: Any = None) -> Any:
        """
        Get value by key.
        
        Args:
            session_id: Session identifier
            key: Key name
            default: Default value if key not found
            
        Returns:
            Stored value or default
        """
        with self._transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM kv_store WHERE session_id = ? AND key = ?",
                (session_id, key)
            )
            row = cursor.fetchone()
            
            if row is None:
                return default
            
            return json.loads(row['value'])
    
    def delete_kv(self, session_id: str, key: str) -> bool:
        """
        Delete key-value pair.
        
        Args:
            session_id: Session identifier
            key: Key name
            
        Returns:
            True if deleted, False if not found
        """
        with self._transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM kv_store WHERE session_id = ? AND key = ?",
                (session_id, key)
            )
            return cursor.rowcount > 0
    
    def get_all_kv(self, session_id: str) -> Dict[str, Any]:
        """
        Get all key-value pairs for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict of all key-value pairs
        """
        with self._transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT key, value FROM kv_store WHERE session_id = ?",
                (session_id,)
            )
            rows = cursor.fetchall()
            
            return {row['key']: json.loads(row['value']) for row in rows}
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite Row to dict with JSON parsing."""
        result = dict(row)
        
        # Parse JSON fields
        for key in ['context_data', 'metadata']:
            if key in result and result[key] is not None:
                try:
                    result[key] = json.loads(result[key])
                except json.JSONDecodeError:
                    pass
        
        return result
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None


# Convenience function for quick testing
def test_persistence():
    """Test the persistence layer."""
    import tempfile
    import os
    
    # Use temporary database for testing
    db_path = tempfile.mktemp(suffix='.db')
    
    try:
        persistence = SQLitePersistence(db_path)
        
        # Test session creation
        print("Testing session creation...")
        success = persistence.create_session(
            session_id="test-session-001",
            agent_id="agent-1",
            context_data={"task": "testing", "priority": "high"},
            metadata={"version": "1.0"}
        )
        print(f"  Created: {success}")
        
        # Test duplicate
        success = persistence.create_session(
            session_id="test-session-001",
            agent_id="agent-1"
        )
        print(f"  Duplicate rejected: {not success}")
        
        # Test get session
        print("\nTesting session retrieval...")
        session = persistence.get_session("test-session-001")
        print(f"  Session ID: {session['session_id']}")
        print(f"  Agent ID: {session['agent_id']}")
        print(f"  Context: {session['context_data']}")
        
        # Test update
        print("\nTesting session update...")
        persistence.update_session(
            "test-session-001",
            context_data={"task": "updated", "progress": 50},
            status="running"
        )
        session = persistence.get_session("test-session-001")
        print(f"  Updated context: {session['context_data']}")
        print(f"  Status: {session['status']}")
        
        # Test steps
        print("\nTesting step operations...")
        step1 = persistence.add_step("test-session-001", "initialize", "success")
        step2 = persistence.add_step("test-session-001", "process_data", "completed 100 items")
        print(f"  Added steps: {step1}, {step2}")
        
        steps = persistence.get_steps("test-session-001")
        print(f"  Retrieved {len(steps)} steps")
        for step in steps:
            print(f"    Step {step['step_number']}: {step['action']} = {step['result']}")
        
        # Test KV store
        print("\nTesting KV store...")
        persistence.set_kv("test-session-001", "counter", 42)
        persistence.set_kv("test-session-001", "config", {"timeout": 30, "retries": 3})
        
        counter = persistence.get_kv("test-session-001", "counter")
        config = persistence.get_kv("test-session-001", "config")
        print(f"  Counter: {counter}")
        print(f"  Config: {config}")
        
        all_kv = persistence.get_all_kv("test-session-001")
        print(f"  All KV pairs: {all_kv}")
        
        # Test list sessions
        print("\nTesting session listing...")
        sessions = persistence.list_sessions()
        print(f"  Total sessions: {len(sessions)}")
        
        # Test delete
        print("\nTesting session deletion...")
        persistence.delete_session("test-session-001")
        session = persistence.get_session("test-session-001")
        print(f"  Session deleted: {session is None}")
        
        print("\n✅ All persistence tests passed!")
        
    finally:
        persistence.close()
        if os.path.exists(db_path):
            os.remove(db_path)


if __name__ == "__main__":
    test_persistence()
