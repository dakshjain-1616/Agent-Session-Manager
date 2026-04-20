"""
Multi-Agent Example for Agent Session Manager

Demonstrates:
- Multiple agents with isolated sessions
- Cross-agent context sharing
- Agent-specific memory retrieval
- Context budget enforcement per agent
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_session_manager import SessionManager


def main():
    """Demonstrate multi-agent capabilities."""
    
    print("=" * 60)
    print("Agent Session Manager - Multi-Agent Example")
    print("=" * 60)
    
    # Initialize manager
    manager = SessionManager(
        db_path="./multi_agent_sessions.db",
        chroma_persist_dir="./multi_agent_chroma",
        default_context_budget=1500
    )
    
    # Define multiple agents
    agents = [
        {
            "agent_id": "research-agent",
            "session_id": "research-session-001",
            "goal": "Research machine learning techniques",
            "tasks": [
                ("Search for transformer architectures", "Found 15 papers"),
                ("Analyze attention mechanisms", "Documented key findings"),
                ("Compare model performance", "BERT vs GPT analysis complete"),
            ]
        },
        {
            "agent_id": "code-agent",
            "session_id": "code-session-001", 
            "goal": "Implement data pipeline",
            "tasks": [
                ("Set up data loaders", "PyTorch DataLoader configured"),
                ("Implement preprocessing", "Tokenization pipeline ready"),
                ("Add augmentation", "Random crop and flip added"),
            ]
        },
        {
            "agent_id": "test-agent",
            "session_id": "test-session-001",
            "goal": "Create test suite",
            "tasks": [
                ("Write unit tests", "85% coverage achieved"),
                ("Set up CI pipeline", "GitHub Actions configured"),
                ("Add integration tests", "End-to-end tests passing"),
            ]
        }
    ]
    
    # Create sessions for each agent
    print("\n🤖 Creating sessions for multiple agents...")
    
    for agent_config in agents:
        agent_id = agent_config["agent_id"]
        session_id = agent_config["session_id"]
        
        # Check if session exists
        existing = manager.list_sessions(agent_id=agent_id)
        session_exists = any(s['session_id'] == session_id for s in existing)
        
        if session_exists:
            print(f"\n   📂 {agent_id}: Session exists, loading...")
            state = manager.load_session(session_id)
            print(f"      Goal: {state.current_goal}")
            print(f"      Steps: {len(state.completed_steps)}")
        else:
            print(f"\n   🆕 {agent_id}: Creating new session...")
            state = manager.create_session(
                session_id=session_id,
                agent_id=agent_id,
                initial_goal=agent_config["goal"],
                metadata={"team": "ml-project", "agent_type": agent_id.split('-')[0]}
            )
            print(f"      Goal: {state.current_goal}")
        
        # Add tasks as steps
        current_steps = len(state.completed_steps)
        tasks = agent_config["tasks"]
        
        if current_steps < len(tasks):
            print(f"      Adding {len(tasks) - current_steps} new tasks...")
            for i, (action, result) in enumerate(tasks[current_steps:], start=current_steps + 1):
                manager.add_step(
                    session_id=session_id,
                    action=action,
                    result=result,
                    metadata={"agent": agent_id}
                )
        else:
            print(f"      All tasks already completed")
        
        # Save state
        manager.save_state(state)
    
    # List all sessions
    print("\n📋 All Sessions:")
    all_sessions = manager.list_sessions()
    for session in all_sessions:
        print(f"   • {session['session_id']} ({session['agent_id']}) - {session['status']}")
    
    # Demonstrate agent-specific memory retrieval
    print("\n🔍 Agent-Specific Memory Retrieval:")
    
    for agent_config in agents:
        agent_id = agent_config["agent_id"]
        session_id = agent_config["session_id"]
        
        print(f"\n   {agent_id}:")
        results = manager.recall_context(
            query="What tasks were completed?",
            session_id=session_id,
            n_results=3
        )
        
        for r in results:
            text = r['text'][:50] + "..." if len(r['text']) > 50 else r['text']
            print(f"      • {text}")
    
    # Demonstrate cross-agent context sharing
    print("\n🌐 Cross-Agent Context Sharing:")
    print("   Searching for 'pipeline' across all agents...")
    
    cross_agent_results = manager.recall_across_agents(
        query="pipeline",
        n_results=5
    )
    
    if cross_agent_results:
        print(f"   Found {len(cross_agent_results)} results:")
        for r in cross_agent_results:
            agent_id = r.get('agent_id', 'unknown')
            text = r['text'][:50] + "..." if len(r['text']) > 50 else r['text']
            print(f"      • [{agent_id}] {text}")
    else:
        print("   No results found")
    
    # Demonstrate context budget per agent
    print("\n📊 Context Budget Per Agent:")
    
    for agent_config in agents:
        agent_id = agent_config["agent_id"]
        session_id = agent_config["session_id"]
        
        state = manager.load_session(session_id)
        budget_info = manager.check_budget(state)
        
        print(f"   {agent_id}:")
        print(f"      Tokens: {budget_info['tokens']} / {budget_info['budget']}")
        print(f"      Usage: {budget_info['usage_percent']:.1f}%")
    
    # Demonstrate budget enforcement with large context
    print("\n⚠️  Testing Budget Enforcement:")
    
    # Create a session with very small budget to trigger trimming
    small_budget_session = "budget-test-session"
    
    try:
        state = manager.create_session(
            session_id=small_budget_session,
            agent_id="budget-test-agent",
            initial_goal="Test budget constraints"
        )
        
        # Add many steps to exceed budget
        print("   Adding steps to exceed budget...")
        for i in range(20):
            manager.add_step(
                session_id=small_budget_session,
                action=f"Step {i+1}: " + "x" * 100,  # Long action text
                result="Result " * 20
            )
        
        # Try to load with small budget
        print("   Loading with 500 token budget...")
        trimmed_state = manager.load_session(
            session_id=small_budget_session,
            context_budget=500
        )
        
        budget_info = manager.check_budget(trimmed_state, budget=500)
        print(f"   Original steps: 20")
        print(f"   Trimmed steps: {len(trimmed_state.completed_steps)}")
        print(f"   Token usage: {budget_info['tokens']} / 500")
        print(f"   Within budget: {'✅' if budget_info['within_budget'] else '❌'}")
        
        # Cleanup test session
        manager.delete_session(small_budget_session)
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Final statistics
    print("\n📈 Final Statistics:")
    
    for agent_config in agents:
        agent_id = agent_config["agent_id"]
        session_id = agent_config["session_id"]
        
        stats = manager.get_session_stats(session_id)
        print(f"   {agent_id}:")
        print(f"      Steps: {stats['step_count']}")
        print(f"      Memory docs: {stats['memory_documents']}")
    
    # Cleanup
    manager.close()
    
    print("\n" + "=" * 60)
    print("✅ Multi-agent demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
