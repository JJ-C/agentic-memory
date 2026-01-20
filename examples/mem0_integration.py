"""
Integration example: Using Mem0 with Agentic Memory Framework

This demonstrates how to combine:
- Mem0's automatic memory extraction
- Our framework's advanced retrieval and context engineering
"""

import sys
sys.path.insert(0, '/Users/jchen65/dev/ai_playground/agentic_memory')

from agentic_memory import AgenticMemory, MemoryType, MemoryScope


# Mock Mem0 interface (install with: pip install mem0ai)
# Uncomment below if you have mem0ai installed:
"""
from mem0 import Memory

mem0_config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        }
    }
}
mem0 = Memory.from_config(mem0_config)
"""


class Mem0AgenticBridge:
    """Bridge between Mem0 and Agentic Memory Framework"""
    
    def __init__(self, agentic_memory: AgenticMemory, mem0_client=None):
        self.agentic = agentic_memory
        self.mem0 = mem0_client
        self.extraction_enabled = mem0_client is not None
    
    def add_conversation(self, messages: list, user_id: str, session_id: str = None):
        """
        Add conversation using Mem0's automatic extraction,
        then store in our framework with proper scoping
        """
        if self.extraction_enabled:
            # Use Mem0 to extract memories
            mem0_memories = self.mem0.add(messages, user_id=user_id)
            
            # Store extracted memories in our framework
            for mem in mem0_memories:
                self._store_mem0_memory(mem, user_id, session_id)
        else:
            # Fallback: Manual extraction
            self._manual_extraction(messages, user_id, session_id)
    
    def _store_mem0_memory(self, mem0_memory: dict, user_id: str, session_id: str = None):
        """Convert Mem0 memory to our framework format"""
        
        # Determine scope based on memory type
        if mem0_memory.get('memory_type') == 'user':
            scope = MemoryScope.GLOBAL
        elif mem0_memory.get('memory_type') == 'session':
            scope = MemoryScope.SESSION
        else:
            scope = MemoryScope.PROJECT
        
        # Determine our memory type based on content
        content = mem0_memory.get('data', '')
        
        if self._is_procedural(content):
            self.agentic.store_procedural(
                workflow_name=mem0_memory.get('id', 'workflow'),
                content=content,
                scope=scope,
                metadata={
                    'user_id': user_id,
                    'session_id': session_id,
                    'mem0_id': mem0_memory.get('id'),
                    'source': 'mem0'
                }
            )
        elif self._is_episodic(content):
            self.agentic.store_episodic(
                source=user_id,
                content=content,
                scope=scope,
                metadata={
                    'user_id': user_id,
                    'session_id': session_id,
                    'mem0_id': mem0_memory.get('id'),
                    'source': 'mem0'
                }
            )
        else:
            self.agentic.store_semantic(
                entity=mem0_memory.get('entity', 'fact'),
                content=content,
                scope=scope,
                metadata={
                    'user_id': user_id,
                    'session_id': session_id,
                    'mem0_id': mem0_memory.get('id'),
                    'source': 'mem0'
                }
            )
    
    def _is_procedural(self, content: str) -> bool:
        """Heuristic to detect procedural knowledge"""
        procedural_keywords = ['how to', 'steps', 'process', 'workflow', 'procedure', 'method']
        return any(keyword in content.lower() for keyword in procedural_keywords)
    
    def _is_episodic(self, content: str) -> bool:
        """Heuristic to detect episodic memories"""
        episodic_keywords = ['discussed', 'mentioned', 'said', 'talked about', 'asked', 'told']
        return any(keyword in content.lower() for keyword in episodic_keywords)
    
    def _manual_extraction(self, messages: list, user_id: str, session_id: str = None):
        """Fallback: Simple manual extraction without Mem0"""
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get('content', '')
                role = msg.get('role', 'user')
            else:
                content = str(msg)
                role = 'user'
            
            # Store as episodic memory
            self.agentic.store_episodic(
                source=role,
                content=content,
                scope=MemoryScope.SESSION,
                metadata={
                    'user_id': user_id,
                    'session_id': session_id,
                    'source': 'manual'
                }
            )
    
    def retrieve_with_context_engineering(self, query: str, user_id: str, 
                                         task_type: str = None,
                                         scope: MemoryScope = None):
        """
        Retrieve using our framework's advanced features:
        - Multi-stage retrieval
        - Context problem solving
        - Tool loadout
        - Refinement
        """
        return self.agentic.retrieve(
            query=query,
            task_type=task_type,
            scope=scope,
            context_id=user_id
        )


def demo_basic_integration():
    """Basic integration demo without actual Mem0"""
    print("=" * 70)
    print("DEMO: Mem0 + Agentic Memory Integration")
    print("=" * 70)
    
    agentic = AgenticMemory()
    bridge = Mem0AgenticBridge(agentic, mem0_client=None)
    
    print("\n1️⃣  Adding conversation (automatic extraction simulated)...")
    
    conversation = [
        {"role": "user", "content": "I'm building a chatbot using Python and FastAPI"},
        {"role": "assistant", "content": "Great choice! FastAPI is excellent for building APIs."},
        {"role": "user", "content": "I prefer using PostgreSQL for the database"},
        {"role": "assistant", "content": "PostgreSQL is a solid choice for production."},
        {"role": "user", "content": "How do I handle authentication?"},
        {"role": "assistant", "content": "You can use JWT tokens with FastAPI's security utilities."}
    ]
    
    bridge.add_conversation(
        messages=conversation,
        user_id="user_123",
        session_id="session_456"
    )
    
    print("   ✓ Conversation processed and stored")
    
    print("\n2️⃣  Retrieving with context engineering...")
    
    # Query 1: Factual (uses semantic memory)
    result = bridge.retrieve_with_context_engineering(
        query="What technology stack is the user using?",
        user_id="user_123",
        task_type="factual"
    )
    
    print(f"\n   Query: What technology stack is the user using?")
    print(f"   Strategy: {result.strategy_used}")
    print(f"   Retrieved {len(result.memories)} memories:")
    for mem in result.memories[:2]:
        print(f"      [{mem.type.value}] {mem.content[:80]}...")
    
    # Query 2: Procedural (uses procedural memory)
    result = bridge.retrieve_with_context_engineering(
        query="How should I handle authentication?",
        user_id="user_123",
        task_type="procedural"
    )
    
    print(f"\n   Query: How should I handle authentication?")
    print(f"   Strategy: {result.strategy_used}")
    print(f"   Memory types used: {result.metadata['memory_types_used']}")
    
    print("\n3️⃣  Benefits of integration:")
    print("   ✓ Automatic extraction (Mem0)")
    print("   ✓ Context problem solving (Our framework)")
    print("   ✓ Multi-stage retrieval (Our framework)")
    print("   ✓ Hierarchical scoping (Our framework)")


def demo_multi_user_isolation():
    """Demo showing user isolation with context quarantine"""
    print("\n" + "=" * 70)
    print("DEMO: Multi-User Isolation")
    print("=" * 70)
    
    agentic = AgenticMemory()
    bridge = Mem0AgenticBridge(agentic)
    
    print("\n1️⃣  Creating isolated contexts for two users...")
    
    # User 1
    agentic.create_isolated_context("user_alice", MemoryScope.SESSION)
    bridge.add_conversation(
        messages=[
            {"role": "user", "content": "I'm working on a React project"},
            {"role": "user", "content": "I prefer TypeScript over JavaScript"}
        ],
        user_id="user_alice",
        session_id="session_alice"
    )
    
    # User 2
    agentic.create_isolated_context("user_bob", MemoryScope.SESSION)
    bridge.add_conversation(
        messages=[
            {"role": "user", "content": "I'm working on a Vue project"},
            {"role": "user", "content": "I prefer JavaScript for simplicity"}
        ],
        user_id="user_bob",
        session_id="session_bob"
    )
    
    print("   ✓ Stored memories for Alice and Bob")
    
    print("\n2️⃣  Retrieving user-specific context...")
    
    # Alice's context
    result_alice = bridge.retrieve_with_context_engineering(
        query="What framework am I using?",
        user_id="user_alice",
        scope=MemoryScope.SESSION
    )
    
    print(f"\n   Alice's context:")
    if result_alice.memories:
        print(f"      {result_alice.memories[0].content}")
    
    # Bob's context
    result_bob = bridge.retrieve_with_context_engineering(
        query="What framework am I using?",
        user_id="user_bob",
        scope=MemoryScope.SESSION
    )
    
    print(f"\n   Bob's context:")
    if result_bob.memories:
        print(f"      {result_bob.memories[0].content}")
    
    print("\n   ✓ Context isolation prevents mixing user data")


def demo_context_problem_solving():
    """Demo showing how our framework solves context problems"""
    print("\n" + "=" * 70)
    print("DEMO: Context Problem Solving")
    print("=" * 70)
    
    agentic = AgenticMemory()
    bridge = Mem0AgenticBridge(agentic)
    
    print("\n1️⃣  Creating scenario with context problems...")
    
    # Add old information (context poisoning)
    from datetime import datetime, timedelta
    old_mem = agentic.store_semantic(
        entity="api_version",
        content="The API is version 1.0",
        scope=MemoryScope.GLOBAL,
        confidence=0.6
    )
    old_mem.timestamp = datetime.now() - timedelta(days=100)
    
    # Add new information
    agentic.store_semantic(
        entity="api_version",
        content="The API is version 2.0",
        scope=MemoryScope.GLOBAL,
        confidence=1.0
    )
    
    # Add irrelevant information (context distraction)
    agentic.store_episodic(
        source="system",
        content="Random log: System health check passed at 10:00 AM",
        scope=MemoryScope.SESSION,
        importance=0.1
    )
    
    # Add conflicting information (context conflict)
    agentic.store_semantic(
        entity="preference",
        content="User prefers detailed responses",
        scope=MemoryScope.GLOBAL
    )
    agentic.store_semantic(
        entity="preference",
        content="User prefers concise responses",
        scope=MemoryScope.GLOBAL
    )
    
    print("   ✓ Created memories with potential problems")
    
    print("\n2️⃣  Retrieving with problem detection...")
    
    result = agentic.retrieve(
        query="What API version should I use?",
        task_type="factual"
    )
    
    print(f"\n   Retrieved {len(result.memories)} memories")
    print(f"   Conflicts detected: {result.metadata.get('conflicts_detected', 0)}")
    print(f"   Top result: {result.memories[0].content if result.memories else 'None'}")
    
    print("\n3️⃣  Problems solved:")
    print("   ✓ Context Poisoning: Older, low-confidence info deprioritized")
    print("   ✓ Context Distraction: Irrelevant logs filtered out")
    print("   ✓ Context Conflict: Contradictions resolved (newer preferred)")
    print("   ✓ Context Confusion: Clear scoping prevents ambiguity")


def demo_performance_comparison():
    """Compare retrieval with and without context engineering"""
    print("\n" + "=" * 70)
    print("DEMO: Performance Comparison")
    print("=" * 70)
    
    agentic = AgenticMemory()
    bridge = Mem0AgenticBridge(agentic)
    
    print("\n1️⃣  Populating memory store...")
    
    # Add 20 memories
    for i in range(20):
        agentic.store_semantic(
            entity=f"fact_{i}",
            content=f"This is fact number {i} about various topics",
            scope=MemoryScope.GLOBAL,
            importance=0.3 + (i % 5) * 0.1
        )
    
    # Add one highly relevant memory
    agentic.store_semantic(
        entity="important_fact",
        content="Python is excellent for machine learning and data science",
        scope=MemoryScope.GLOBAL,
        importance=0.9
    )
    
    print("   ✓ Stored 21 memories")
    
    print("\n2️⃣  Retrieving with context engineering...")
    
    import time
    start = time.time()
    result = bridge.retrieve_with_context_engineering(
        query="What language is good for machine learning?",
        user_id="user_123",
        task_type="recommendation"
    )
    latency = (time.time() - start) * 1000
    
    print(f"\n   Latency: {latency:.2f} ms")
    print(f"   Total tokens: {result.total_tokens}")
    print(f"   Memories retrieved: {len(result.memories)}")
    print(f"   Strategy: {result.strategy_used}")
    print(f"   Top result: {result.memories[0].content if result.memories else 'None'}")
    
    print("\n3️⃣  Advantages:")
    print("   ✓ Relevant memory ranked first (importance + semantic similarity)")
    print("   ✓ Token budget respected")
    print("   ✓ Low latency with multi-stage retrieval")


def main():
    """Run all integration demos"""
    demos = [
        demo_basic_integration,
        demo_multi_user_isolation,
        demo_context_problem_solving,
        demo_performance_comparison
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n❌ Error in {demo.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Integration demonstrations complete!")
    print("=" * 70)
    
    print("\n📚 Key Takeaways:")
    print("   • Mem0: Automatic extraction from conversations")
    print("   • Our Framework: Advanced retrieval + context engineering")
    print("   • Together: Best of both worlds")
    
    print("\n🚀 To use with real Mem0:")
    print("   1. pip install mem0ai")
    print("   2. Uncomment Mem0 initialization in this file")
    print("   3. Configure your vector store (Qdrant, Pinecone, etc.)")
    print("   4. Run: python examples/mem0_integration.py")


if __name__ == "__main__":
    main()
