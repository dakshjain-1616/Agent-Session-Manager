"""
Basic Usage Example for Agent Session Manager

Demonstrates:
- Session creation and persistence
- Adding steps and state tracking
- Session restoration across script runs
- Semantic recall for context retrieval
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_session_manager import SessionManager


def main():
    """Demonstrate basic usage of the SessionManager."""
    
    print("=" * 60)
    print("Agent Session Manager - Basic Usage Example")
    print("=" * 60)
    
    # Initialize the manager
    # This creates SQLite and ChromaDB connections
    manager = SessionManager(
        db_path="./demo_sessions.db",
        chroma_persist_dir="./demo_chroma",
        default_context_budget=2000
    )
    
    session_id = "demo-session-001"
    
    # Check if session already exists (from previous run)
    existing_sessions = manager.list_sessions()
    session_exists = any(s['session_id'] == session_id for s in existing_sessions)
    
    if session_exists:
        print("\n📂 Session exists from previous run - loading it...")
        state = manager.load_session(session_id)
        print(f"   Loaded session: {state.session_id}")
        print(f"   Agent ID: {state.agent_id}")
        print(f"   Current goal: {state.current_goal}")
        print(f"   Completed steps: {len(state.completed_steps)}")
        
        # Show previous steps
        if state.completed_steps:
            print("\n   Previous steps:")
            for step in state.completed_steps[-3:]:  # Show last 3
                print(f"     {step['step_number']}. {step['action']}")
                if step.get('result'):
                    print(f"        → {step['result']}")
    else:
        print("\n🆕 Creating new session...")
        state = manager.create_session(
            session_id=session_id,
            agent_id="demo-agent",
            initial_goal="Build a semantic search system",
            metadata={
                "project": "demo",
                "priority": "high",
                "tags": ["search", "nlp"]
            }
        )
        print(f"   Created session: {state.session_id}")
        print(f"   Agent ID: {state.agent_id}")
        print(f"   Initial goal: {state.current_goal}")
    
    # Add some steps to demonstrate persistence
    print("\n📝 Adding new steps...")
    
    steps_to_add = [
        ("Initialize ChromaDB client", "Connected to local storage"),
        ("Load embedding model", "Loaded all-MiniLM-L6-v2"),
        ("Index sample documents", "Indexed 10 documents"),
        ("Test semantic search", "Retrieved relevant results"),
    ]
    
    # Only add steps if we haven't added them before
    current_step_count = len(state.completed_steps)
    if current_step_count < len(steps_to_add):
        for i, (action, result) in enumerate(steps_to_add[current_step_count:], start=current_step_count + 1):
            step_num = manager.add_step(
                session_id=session_id,
                action=action,
                result=result,
                metadata={"demo": True, "iteration": i}
            )
            print(f"   Added step {step_num}: {action}")
    else:
        print("   All steps already added in previous run")
    
    # Save the current state
    print("\n💾 Saving state...")
    manager.save_state(state)
    print("   State saved to SQLite and indexed in ChromaDB")
    
    # Demonstrate semantic recall
    print("\n🔍 Testing semantic recall...")
    
    queries = [
        "What embedding model was used?",
        "Tell me about the search functionality",
        "What steps were completed?"
    ]
    
    for query in queries:
        print(f"\n   Query: \"{query}\"")
        results = manager.recall_context(
            query=query,
            session_id=session_id,
            n_results=2
        )
        
        if results:
            print(f"   Found {len(results)} relevant memories:")
            for r in results:
                text = r['text'][:60] + "..." if len(r['text']) > 60 else r['text']
                print(f"     • {text} (relevance: {1 - r['distance']:.2f})")
        else:
            print("   No relevant memories found")
    
    # Check context budget
    print("\n📊 Checking context budget...")
    state = manager.load_session(session_id)  # Reload to get updated steps
    budget_info = manager.check_budget(state)
    print(f"   Token usage: {budget_info['tokens']} / {budget_info['budget']}")
    print(f"   Usage: {budget_info['usage_percent']:.1f}%")
    print(f"   Within budget: {'✅' if budget_info['within_budget'] else '❌'}")
    
    # Get session statistics
    print("\n📈 Session Statistics:")
    stats = manager.get_session_stats(session_id)
    print(f"   Session ID: {stats['session_id']}")
    print(f"   Agent ID: {stats['agent_id']}")
    print(f"   Status: {stats['status']}")
    print(f"   Total steps: {stats['step_count']}")
    print(f"   Memory documents: {stats['memory_documents']}")
    print(f"   Created: {stats['created_at']}")
    print(f"   Last updated: {stats['updated_at']}")
    
    # Cleanup
    manager.close()
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("=" * 60)
    print("\nRun this script again to see persistence in action.")
    print("The session will be restored from the previous run.")


if __name__ == "__main__":
    main()
