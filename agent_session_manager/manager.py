"""
Core Session Manager for Agent Session Manager.

Orchestrates SQLite persistence and ChromaDB semantic memory,
providing context budget management and multi-agent support.
"""

import os
import json
import time
import tiktoken
from typing import Optional, Dict, List, Any, Callable, Union
from datetime import datetime
from dataclasses import dataclass, asdict

from .persistence import SQLitePersistence
from .memory import ChromaDBMemory


class SessionManagerError(Exception):
    """Base exception for SessionManager errors."""
    pass


class SessionNotFoundError(SessionManagerError):
    """Raised when a session is not found."""
    pass


class BudgetExceededError(SessionManagerError):
    """Raised when context budget is exceeded."""
    pass


@dataclass
class AgentState:
    """Represents the state of an agent session."""
    session_id: str
    agent_id: str
    current_goal: Optional[str] = None
    completed_steps: List[Dict[str, Any]] = None
    pending_steps: List[str] = None
    error_history: List[Dict[str, Any]] = None
    tool_outputs: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.completed_steps is None:
            self.completed_steps = []
        if self.pending_steps is None:
            self.pending_steps = []
        if self.error_history is None:
            self.error_history = []
        if self.tool_outputs is None:
            self.tool_outputs = {}
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        """Create state from dictionary."""
        return cls(**data)


class SessionManager:
    """
    Main API for managing agent sessions with persistence and semantic memory.
    
    Features:
    - State persistence via SQLite
    - Semantic recall via ChromaDB
    - Context budget management with intelligent trimming
    - Multi-agent support with namespace isolation
    """
    
    def __init__(
        self,
        db_path: str = "agent_sessions.db",
        chroma_persist_dir: str = "./chroma_db",
        collection_name: str = "agent_memory",
        embedding_model: str = "all-MiniLM-L6-v2",
        default_context_budget: int = 4000,
        tokenizer_model: str = "cl100k_base"
    ):
        """
        Initialize SessionManager.
        
        Args:
            db_path: Path to SQLite database
            chroma_persist_dir: Directory for ChromaDB persistence
            collection_name: Name of ChromaDB collection
            embedding_model: Sentence-transformers model name
            default_context_budget: Default token budget for context
            tokenizer_model: Tiktoken model for token counting
        """
        self.db_path = db_path
        self.chroma_persist_dir = chroma_persist_dir
        self.collection_name = collection_name
        self.default_context_budget = default_context_budget
        self.tokenizer_model = tokenizer_model
        
        # Initialize components
        self._persistence = SQLitePersistence(db_path)
        self._memory = ChromaDBMemory(
            collection_name=collection_name,
            persist_directory=chroma_persist_dir,
            embedding_model=embedding_model
        )
        
        # Initialize tokenizer
        try:
            self._tokenizer = tiktoken.get_encoding(tokenizer_model)
        except Exception:
            # Fallback to cl100k_base if model not found
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def create_session(
        self,
        session_id: str,
        agent_id: str,
        initial_goal: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """
        Create a new agent session.
        
        Args:
            session_id: Unique session identifier
            agent_id: Agent identifier
            initial_goal: Initial goal for the session
            metadata: Additional session metadata
            
        Returns:
            AgentState object
            
        Raises:
            SessionManagerError: If session already exists
        """
        # Check if session exists
        existing = self._persistence.get_session(session_id)
        if existing:
            raise SessionManagerError(f"Session '{session_id}' already exists")
        
        # Create session in SQLite
        context_data = {
            "current_goal": initial_goal,
            "created_at": datetime.now().isoformat()
        }
        
        success = self._persistence.create_session(
            session_id=session_id,
            agent_id=agent_id,
            context_data=context_data,
            metadata=metadata or {}
        )
        
        if not success:
            raise SessionManagerError(f"Failed to create session '{session_id}'")
        
        # Create initial state
        state = AgentState(
            session_id=session_id,
            agent_id=agent_id,
            current_goal=initial_goal,
            metadata=metadata or {}
        )
        
        # Store initial goal in memory if provided
        if initial_goal:
            self._memory.add_document(
                text=f"Session started with goal: {initial_goal}",
                metadata={"type": "goal", "agent_id": agent_id},
                session_id=session_id
            )
        
        return state
    
    def load_session(
        self,
        session_id: str,
        context_budget: Optional[int] = None
    ) -> AgentState:
        """
        Load a session with context trimming to stay within budget.
        
        Args:
            session_id: Session identifier
            context_budget: Token budget (uses default if not specified)
            
        Returns:
            AgentState with trimmed context
            
        Raises:
            SessionNotFoundError: If session not found
        """
        # Get session from SQLite
        session_data = self._persistence.get_session(session_id)
        if not session_data:
            raise SessionNotFoundError(f"Session '{session_id}' not found")
        
        # Get steps
        steps = self._persistence.get_steps(session_id)
        
        # Get KV store data
        kv_data = self._persistence.get_all_kv(session_id)
        
        # Build state
        state = AgentState(
            session_id=session_id,
            agent_id=session_data['agent_id'],
            current_goal=session_data.get('context_data', {}).get('current_goal'),
            metadata={**session_data.get('metadata', {}), **kv_data}
        )
        
        # Process steps into completed/pending
        for step in steps:
            step_data = {
                "step_number": step['step_number'],
                "action": step['action'],
                "result": step['result'],
                "timestamp": step['timestamp'],
                "metadata": step.get('metadata', {})
            }
            state.completed_steps.append(step_data)
        
        # Apply context budget trimming
        budget = context_budget or self.default_context_budget
        state = self._trim_context(state, budget)
        
        return state
    
    def save_state(self, state: AgentState, persist_to_memory: bool = True) -> None:
        """
        Save agent state to persistence.
        
        Args:
            state: AgentState to save
            persist_to_memory: Whether to also index in semantic memory
        """
        # Update session context
        context_data = {
            "current_goal": state.current_goal,
            "updated_at": datetime.now().isoformat()
        }
        
        self._persistence.update_session(
            session_id=state.session_id,
            context_data=context_data,
            metadata=state.metadata
        )
        
        # Save KV store data
        for key, value in state.metadata.items():
            self._persistence.set_kv(state.session_id, key, value)
        
        # Index in semantic memory if requested
        if persist_to_memory and state.completed_steps:
            # Index recent steps
            recent_steps = state.completed_steps[-5:]  # Last 5 steps
            for step in recent_steps:
                text = f"Step {step['step_number']}: {step['action']}"
                if step.get('result'):
                    text += f" -> {step['result']}"
                
                self._memory.add_document(
                    text=text,
                    metadata={
                        "type": "step",
                        "step_number": step['step_number'],
                        "agent_id": state.agent_id
                    },
                    session_id=state.session_id
                )
        
        # Index goal if present
        if state.current_goal and persist_to_memory:
            self._memory.add_document(
                text=f"Current goal: {state.current_goal}",
                metadata={"type": "goal", "agent_id": state.agent_id},
                session_id=state.session_id
            )
    
    def add_step(
        self,
        session_id: str,
        action: str,
        result: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        index_in_memory: bool = True
    ) -> int:
        """
        Add a step to a session.
        
        Args:
            session_id: Session identifier
            action: Action description
            result: Action result
            metadata: Step metadata
            index_in_memory: Whether to index in semantic memory
            
        Returns:
            Step number assigned
        """
        # Add to SQLite
        step_number = self._persistence.add_step(
            session_id=session_id,
            action=action,
            result=result,
            metadata=metadata
        )
        
        # Index in semantic memory
        if index_in_memory:
            text = f"Step {step_number}: {action}"
            if result:
                text += f" -> {result}"
            
            self._memory.add_document(
                text=text,
                metadata={
                    "type": "step",
                    "step_number": step_number
                },
                session_id=session_id
            )
        
        return step_number
    
    def recall_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        n_results: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically relevant past context.
        
        Args:
            query: Query text for semantic search
            session_id: Optional session filter
            n_results: Number of results to return
            filter_dict: Additional metadata filters
            
        Returns:
            List of relevant context items
        """
        results = self._memory.search_similar(
            query_text=query,
            n_results=n_results,
            filter_dict=filter_dict,
            session_id=session_id
        )
        
        return results
    
    def recall_across_agents(
        self,
        query: str,
        agent_ids: Optional[List[str]] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for context across multiple agents.
        
        Args:
            query: Query text
            agent_ids: List of agent IDs to search (None = all agents)
            n_results: Number of results per agent
            
        Returns:
            List of relevant context items with agent_id
        """
        all_results = []
        
        if agent_ids:
            for agent_id in agent_ids:
                # Get sessions for this agent
                sessions = self._persistence.list_sessions(agent_id=agent_id)
                for session in sessions:
                    results = self._memory.search_similar(
                        query_text=query,
                        n_results=n_results,
                        session_id=session['session_id']
                    )
                    for r in results:
                        r['agent_id'] = agent_id
                    all_results.extend(results)
        else:
            # Search all sessions
            results = self._memory.search_similar(
                query_text=query,
                n_results=n_results
            )
            all_results.extend(results)
        
        # Sort by distance (relevance) and return top results
        all_results.sort(key=lambda x: x.get('distance', float('inf')))
        return all_results[:n_results]
    
    def list_sessions(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all sessions with optional filtering.
        
        Args:
            agent_id: Filter by agent ID
            status: Filter by status
            
        Returns:
            List of session data
        """
        return self._persistence.list_sessions(agent_id=agent_id, status=status)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all associated data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        # Delete from SQLite
        deleted = self._persistence.delete_session(session_id)
        
        # Delete from ChromaDB
        if deleted:
            self._memory.delete_session_documents(session_id)
        
        return deleted
    
    def check_budget(
        self,
        state: AgentState,
        budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Check current context usage against budget.
        
        Args:
            state: AgentState to check
            budget: Token budget (uses default if not specified)
            
        Returns:
            Dict with token count, budget, and usage percentage
        """
        budget = budget or self.default_context_budget
        
        # Serialize state to JSON for token counting
        state_json = json.dumps(state.to_dict())
        tokens = len(self._tokenizer.encode(state_json))
        
        return {
            "tokens": tokens,
            "budget": budget,
            "usage_percent": (tokens / budget) * 100,
            "remaining": budget - tokens,
            "within_budget": tokens <= budget
        }
    
    def _trim_context(
        self,
        state: AgentState,
        budget: int
    ) -> AgentState:
        """
        Trim context to stay within token budget.
        
        Strategy:
        1. Always keep most recent steps
        2. Remove older steps if over budget
        3. Keep metadata and current goal
        
        Args:
            state: AgentState to trim
            budget: Token budget
            
        Returns:
            Trimmed AgentState
        """
        # Create a copy to avoid modifying original
        trimmed_state = AgentState(
            session_id=state.session_id,
            agent_id=state.agent_id,
            current_goal=state.current_goal,
            completed_steps=list(state.completed_steps),
            pending_steps=list(state.pending_steps),
            error_history=list(state.error_history),
            tool_outputs=dict(state.tool_outputs),
            metadata=dict(state.metadata)
        )
        
        # Check current usage
        budget_info = self.check_budget(trimmed_state, budget)
        
        if budget_info["within_budget"]:
            return trimmed_state
        
        # Trim steps from oldest first
        while not self.check_budget(trimmed_state, budget)["within_budget"] and trimmed_state.completed_steps:
            # Remove oldest step
            trimmed_state.completed_steps.pop(0)
        
        # If still over budget, trim metadata
        if not self.check_budget(trimmed_state, budget)["within_budget"]:
            # Keep only essential metadata
            essential_keys = ['agent_id', 'session_id', 'created_at']
            trimmed_state.metadata = {k: v for k, v in trimmed_state.metadata.items() 
                                      if k in essential_keys}
        
        return trimmed_state
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict with session statistics
        """
        session = self._persistence.get_session(session_id)
        if not session:
            raise SessionNotFoundError(f"Session '{session_id}' not found")
        
        step_count = self._persistence.get_step_count(session_id)
        memory_count = self._memory.count_documents(session_id)
        
        return {
            "session_id": session_id,
            "agent_id": session['agent_id'],
            "created_at": session['created_at'],
            "updated_at": session['updated_at'],
            "status": session['status'],
            "step_count": step_count,
            "memory_documents": memory_count
        }
    
    def close(self):
        """Close all connections and cleanup resources."""
        self._persistence.close()


def test_session_manager():
    """Test the SessionManager."""
    import tempfile
    import shutil
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    chroma_dir = os.path.join(temp_dir, "chroma")
    
    try:
        print("Initializing SessionManager...")
        manager = SessionManager(
            db_path=db_path,
            chroma_persist_dir=chroma_dir,
            default_context_budget=1000
        )
        
        # Test session creation
        print("\nTesting session creation...")
        state = manager.create_session(
            session_id="test-session-001",
            agent_id="agent-1",
            initial_goal="Build a chatbot",
            metadata={"project": "demo"}
        )
        print(f"  Created session: {state.session_id}")
        print(f"  Agent: {state.agent_id}")
        print(f"  Goal: {state.current_goal}")
        
        # Test adding steps
        print("\nTesting step addition...")
        for i in range(3):
            step_num = manager.add_step(
                session_id="test-session-001",
                action=f"Action {i+1}",
                result=f"Result {i+1}",
                metadata={"iteration": i+1}
            )
            print(f"  Added step {step_num}")
        
        # Test state saving
        print("\nTesting state saving...")
        state.completed_steps.append({
            "step_number": 4,
            "action": "Final action",
            "result": "Success"
        })
        manager.save_state(state)
        print("  State saved")
        
        # Test session loading
        print("\nTesting session loading...")
        loaded_state = manager.load_session("test-session-001")
        print(f"  Loaded session: {loaded_state.session_id}")
        print(f"  Steps: {len(loaded_state.completed_steps)}")
        
        # Test budget checking
        print("\nTesting budget checking...")
        budget_info = manager.check_budget(loaded_state)
        print(f"  Tokens: {budget_info['tokens']}")
        print(f"  Budget: {budget_info['budget']}")
        print(f"  Usage: {budget_info['usage_percent']:.1f}%")
        
        # Test semantic recall
        print("\nTesting semantic recall...")
        results = manager.recall_context(
            query="What actions were performed?",
            session_id="test-session-001",
            n_results=3
        )
        print(f"  Found {len(results)} relevant documents")
        for r in results:
            print(f"    - {r['text'][:50]}... (distance: {r['distance']:.4f})")
        
        # Test session listing
        print("\nTesting session listing...")
        sessions = manager.list_sessions()
        print(f"  Total sessions: {len(sessions)}")
        
        # Test stats
        print("\nTesting session stats...")
        stats = manager.get_session_stats("test-session-001")
        print(f"  Steps: {stats['step_count']}")
        print(f"  Memory docs: {stats['memory_documents']}")
        
        print("\n✅ All SessionManager tests passed!")
        
    finally:
        manager.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_session_manager()
