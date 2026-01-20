import sys
sys.path.insert(0, '/Users/jchen65/dev/ai_playground/agentic_memory')

from agentic_memory import AgenticMemory, MemoryType, MemoryScope
import json


def demo_basic_usage():
    """Basic usage: Store and retrieve memories"""
    print("=" * 60)
    print("DEMO 1: Basic Usage")
    print("=" * 60)
    
    memory = AgenticMemory()
    
    memory.store_episodic(
        source="user",
        content="I'm working on a Python web application using FastAPI.",
        scope=MemoryScope.SESSION
    )
    
    memory.store_semantic(
        entity="user_preference",
        content="User prefers Python for backend development.",
        scope=MemoryScope.GLOBAL
    )
    
    result = memory.retrieve(
        query="What programming language should I use?",
        task_type="recommendation"
    )
    
    print(f"\nQuery: What programming language should I use?")
    print(f"Retrieved {len(result.memories)} memories:")
    for i, mem in enumerate(result.memories, 1):
        print(f"\n{i}. [{mem.type.value}] {mem.content}")
        print(f"   Relevance: {result.relevance_scores.get(mem.id, 0):.3f}")
    
    print(f"\nTotal tokens: {result.total_tokens}")
    print(f"Strategy: {result.strategy_used}")


def demo_customer_support():
    """Real-world: Customer support with context isolation"""
    print("\n" + "=" * 60)
    print("DEMO 2: Customer Support with Context Isolation")
    print("=" * 60)
    
    memory = AgenticMemory()
    
    memory.create_isolated_context("customer_alice", MemoryScope.SESSION)
    memory.create_isolated_context("customer_bob", MemoryScope.SESSION)
    
    memory.store_episodic(
        source="customer_alice",
        content="Alice reported that she cannot access her account. Email: alice@example.com. Issue: Forgot password.",
        scope=MemoryScope.SESSION,
        tags=["customer_alice", "account_access"]
    )
    
    memory.store_episodic(
        source="customer_bob",
        content="Bob is experiencing slow page load times. Browser: Chrome. Location: US-West.",
        scope=MemoryScope.SESSION,
        tags=["customer_bob", "performance"]
    )
    
    print("\n--- Agent handling Alice's query ---")
    result_alice = memory.retrieve(
        query="What is the customer's issue?",
        context_id="customer_alice",
        scope=MemoryScope.SESSION
    )
    
    print(f"Retrieved for Alice: {result_alice.memories[0].content}")
    
    print("\n--- Agent handling Bob's query ---")
    result_bob = memory.retrieve(
        query="What is the customer's issue?",
        context_id="customer_bob",
        scope=MemoryScope.SESSION
    )
    
    print(f"Retrieved for Bob: {result_bob.memories[0].content}")
    
    print("\n✓ Context isolation prevents mixing customer data")


def demo_code_assistant():
    """Real-world: Code assistant with project-specific context"""
    print("\n" + "=" * 60)
    print("DEMO 3: Code Assistant with Project Context")
    print("=" * 60)
    
    memory = AgenticMemory()
    
    memory.store_semantic(
        entity="project_config",
        content="Project uses React 18, TypeScript, TailwindCSS, and Vite for bundling.",
        scope=MemoryScope.PROJECT,
        tags=["tech_stack"]
    )
    
    memory.store_procedural(
        workflow_name="component_creation",
        content="When creating components: 1) Create in src/components, 2) Use functional components with TypeScript, 3) Add PropTypes, 4) Export as default.",
        scope=MemoryScope.PROJECT,
        tags=["best_practice"]
    )
    
    memory.store_episodic(
        source="developer",
        content="Fixed CORS issue by adding proxy configuration in vite.config.ts",
        scope=MemoryScope.PROJECT,
        importance=0.8,
        tags=["debugging", "cors"]
    )
    
    print("\n--- Query: How should I create a new component? ---")
    result = memory.retrieve(
        query="How should I create a new component?",
        task_type="procedural",
        scope=MemoryScope.PROJECT
    )
    
    print(f"\nMemory types used: {result.metadata['memory_types_used']}")
    for mem in result.memories:
        print(f"\n[{mem.type.value}] {mem.content}")
    
    print("\n--- Query: How was the CORS issue fixed? ---")
    result = memory.retrieve(
        query="How was the CORS issue fixed?",
        task_type="debugging",
        scope=MemoryScope.PROJECT
    )
    
    for mem in result.memories:
        print(f"\n[{mem.type.value}] {mem.content}")


def demo_context_distraction():
    """Demonstrate filtering out irrelevant context"""
    print("\n" + "=" * 60)
    print("DEMO 4: Context Distraction Prevention")
    print("=" * 60)
    
    memory = AgenticMemory()
    
    print("\n--- Storing 15 low-importance memories (noise) ---")
    for i in range(15):
        memory.store_episodic(
            source="system",
            content=f"Routine log entry {i}: System health check passed.",
            scope=MemoryScope.SESSION,
            importance=0.1
        )
    
    print("--- Storing 1 high-importance memory (signal) ---")
    memory.store_episodic(
        source="system",
        content="CRITICAL: Database connection pool exhausted. Increased max connections from 50 to 100.",
        scope=MemoryScope.SESSION,
        importance=0.95,
        tags=["critical", "database"]
    )
    
    print("\n--- Retrieving with query: 'What critical issues occurred?' ---")
    result = memory.retrieve(
        query="What critical issues occurred?",
        task_type="debugging",
        top_k=3
    )
    
    print(f"\nRetrieved {len(result.memories)} memories (top 3):")
    for i, mem in enumerate(result.memories, 1):
        print(f"\n{i}. Importance: {mem.importance:.2f}")
        print(f"   Content: {mem.content[:80]}...")
    
    print(f"\n✓ High-importance memory ranked first despite noise")


def demo_conflict_resolution():
    """Demonstrate handling contradictory information"""
    print("\n" + "=" * 60)
    print("DEMO 5: Context Clash Resolution")
    print("=" * 60)
    
    memory = AgenticMemory()
    
    print("\n--- Storing conflicting preferences ---")
    mem1 = memory.store_semantic(
        entity="user_preference",
        content="User prefers verbose explanations with detailed examples.",
        scope=MemoryScope.GLOBAL,
        importance=0.7
    )
    print(f"Memory 1 (older): {mem1.content}")
    
    import time
    time.sleep(0.1)
    
    mem2 = memory.store_semantic(
        entity="user_preference",
        content="User prefers concise responses without too much detail.",
        scope=MemoryScope.GLOBAL,
        importance=0.7
    )
    print(f"Memory 2 (newer): {mem2.content}")
    
    print("\n--- Retrieving: 'How does the user prefer responses?' ---")
    result = memory.retrieve(
        query="How does the user prefer responses?",
        scope=MemoryScope.GLOBAL
    )
    
    print(f"\nConflicts detected: {result.metadata.get('conflicts_detected', 0)}")
    print(f"Resolved to most recent memory:")
    print(f"  {result.memories[0].content}")
    print(f"  Timestamp: {result.memories[0].timestamp}")


def demo_hierarchical_scope():
    """Demonstrate scope hierarchy and access control"""
    print("\n" + "=" * 60)
    print("DEMO 6: Hierarchical Scope Access")
    print("=" * 60)
    
    memory = AgenticMemory()
    
    memory.store_semantic(
        entity="company_policy",
        content="All code must pass security review before deployment.",
        scope=MemoryScope.GLOBAL,
        importance=1.0
    )
    
    memory.store_semantic(
        entity="project_guideline",
        content="This project uses microservices architecture with Docker.",
        scope=MemoryScope.PROJECT,
        importance=0.8
    )
    
    memory.store_episodic(
        source="developer",
        content="Today's task: Implement user authentication service.",
        scope=MemoryScope.SESSION,
        importance=0.6
    )
    
    memory.store_episodic(
        source="developer",
        content="Current subtask: Writing unit tests for login endpoint.",
        scope=MemoryScope.TASK,
        importance=0.5
    )
    
    print("\n--- Query from TASK scope ---")
    result = memory.retrieve(
        query="What should I be working on?",
        scope=MemoryScope.TASK
    )
    
    print(f"Accessible scopes from TASK level:")
    scopes = {m.scope.value for m in result.memories}
    print(f"  {sorted(scopes)}")
    
    print("\n--- Query from SESSION scope ---")
    result = memory.retrieve(
        query="What guidelines apply?",
        scope=MemoryScope.SESSION
    )
    
    print(f"Accessible scopes from SESSION level:")
    scopes = {m.scope.value for m in result.memories}
    print(f"  {sorted(scopes)}")


def demo_statistics_and_management():
    """Demonstrate memory statistics and management"""
    print("\n" + "=" * 60)
    print("DEMO 7: Memory Statistics and Management")
    print("=" * 60)
    
    memory = AgenticMemory(enable_auto_pruning=False)
    
    for i in range(5):
        memory.store_episodic(
            source="user",
            content=f"Episodic memory {i}",
            scope=MemoryScope.SESSION
        )
    
    for i in range(3):
        memory.store_semantic(
            entity=f"fact_{i}",
            content=f"Semantic memory {i}",
            scope=MemoryScope.GLOBAL
        )
    
    for i in range(2):
        memory.store_procedural(
            workflow_name=f"workflow_{i}",
            content=f"Procedural memory {i}",
            scope=MemoryScope.PROJECT
        )
    
    stats = memory.get_statistics()
    
    print("\n--- Memory Statistics ---")
    print(json.dumps(stats, indent=2))
    
    print("\n--- Pruning low-importance memories ---")
    for i in range(5):
        memory.store_episodic(
            source="test",
            content=f"Low importance memory {i}",
            scope=MemoryScope.SESSION,
            importance=0.05
        )
    
    pruned = memory.prune_old_memories(days_threshold=0, importance_threshold=0.1)
    print(f"Pruned {pruned} memories")
    
    stats_after = memory.get_statistics()
    print(f"\nMemories before: {stats['total_memories']}")
    print(f"Memories after: {stats_after['total_memories']}")


def demo_export_import():
    """Demonstrate memory persistence"""
    print("\n" + "=" * 60)
    print("DEMO 8: Memory Export and Import")
    print("=" * 60)
    
    memory1 = AgenticMemory()
    
    memory1.store_semantic(
        entity="important_fact",
        content="The answer to life, the universe, and everything is 42.",
        scope=MemoryScope.GLOBAL,
        importance=1.0
    )
    
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        filepath = f.name
    
    print(f"\n--- Exporting memories to {filepath} ---")
    memory1.export_to_file(filepath)
    print("✓ Export complete")
    
    memory2 = AgenticMemory()
    
    print(f"\n--- Importing memories from {filepath} ---")
    memory2.import_from_file(filepath)
    print("✓ Import complete")
    
    result = memory2.retrieve(
        query="What is the answer to life?",
        scope=MemoryScope.GLOBAL
    )
    
    print(f"\nRetrieved from imported memory:")
    print(f"  {result.memories[0].content}")
    
    os.unlink(filepath)


def main():
    """Run all demonstrations"""
    demos = [
        demo_basic_usage,
        demo_customer_support,
        demo_code_assistant,
        demo_context_distraction,
        demo_conflict_resolution,
        demo_hierarchical_scope,
        demo_statistics_and_management,
        demo_export_import
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n❌ Error in {demo.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("All demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
