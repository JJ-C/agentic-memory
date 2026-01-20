import sys
sys.path.insert(0, '/Users/jchen65/dev/ai_playground/agentic_memory')

from agentic_memory import AgenticMemory, MemoryScope


def quickstart():
    """Quick start guide for the Agentic Memory Framework"""
    
    print("🚀 Agentic Memory Framework - Quick Start\n")
    
    memory = AgenticMemory()
    
    print("1️⃣  Storing memories...")
    
    memory.store_episodic(
        source="user",
        content="I'm building a chatbot for customer support.",
        scope=MemoryScope.SESSION
    )
    
    memory.store_semantic(
        entity="user_preference",
        content="User prefers Python and FastAPI for backend development.",
        scope=MemoryScope.GLOBAL
    )
    
    memory.store_procedural(
        workflow_name="debugging_process",
        content="When debugging: 1) Check logs, 2) Reproduce issue, 3) Add tests, 4) Fix and verify.",
        scope=MemoryScope.PROJECT
    )
    
    print("   ✓ Stored 3 memories (episodic, semantic, procedural)\n")
    
    print("2️⃣  Retrieving relevant context...")
    
    result = memory.retrieve(
        query="What technology stack should I use?",
        task_type="recommendation"
    )
    
    print(f"   ✓ Retrieved {len(result.memories)} relevant memories")
    print(f"   ✓ Total tokens: {result.total_tokens}")
    print(f"   ✓ Strategy: {result.strategy_used}\n")
    
    print("3️⃣  Retrieved content:")
    for i, mem in enumerate(result.memories, 1):
        print(f"\n   [{mem.type.value}] {mem.content}")
    
    print("\n" + "=" * 60)
    
    print("\n4️⃣  Context isolation example...")
    
    memory.create_isolated_context("project_a", MemoryScope.SESSION)
    
    memory.store_episodic(
        source="project_a",
        content="Project A uses React and TypeScript.",
        scope=MemoryScope.SESSION,
        tags=["project_a"]
    )
    
    result = memory.retrieve(
        query="What framework does the project use?",
        context_id="project_a",
        scope=MemoryScope.SESSION
    )
    
    print(f"   ✓ Context isolated to project_a")
    print(f"   ✓ Retrieved: {result.memories[0].content}\n")
    
    print("=" * 60)
    print("\n✅ Quick start complete!")
    print("\nNext steps:")
    print("  • Run 'python examples/demo.py' for comprehensive demos")
    print("  • Run 'pytest tests/test_scenarios.py' for real-world scenarios")
    print("  • Check README.md for detailed documentation")


if __name__ == "__main__":
    quickstart()
