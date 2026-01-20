# Mem0 Integration Guide

## Overview

This guide shows how to integrate Mem0's automatic memory extraction with our Agentic Memory Framework's advanced retrieval and context engineering capabilities.

## Why Integrate?

### Mem0 Strengths
- ✅ Automatic memory extraction from conversations
- ✅ Simple API for adding memories
- ✅ Built-in LLM integration
- ✅ Managed service option

### Our Framework Strengths
- ✅ Solves 4 context engineering problems
- ✅ Hierarchical scoping (Global → Project → Session → Task)
- ✅ Context quarantine and isolation
- ✅ Multi-stage retrieval with tool loadout
- ✅ Adaptive refinement (pruning + summarization)
- ✅ Comprehensive evaluation framework

### Together
**Mem0 extracts → Our framework retrieves with context engineering**

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Conversation                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                 Mem0 (Extraction Layer)                  │
│  • Automatic fact extraction                            │
│  • Entity recognition                                   │
│  • Memory deduplication                                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Mem0-Agentic Bridge (Adapter)              │
│  • Convert Mem0 format → Our format                     │
│  • Map memory types                                     │
│  • Apply proper scoping                                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│           Agentic Memory (Retrieval Layer)              │
│  • Multi-stage retrieval                                │
│  • Context problem solving                              │
│  • Tool loadout                                         │
│  • Refinement                                           │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    Optimized Context                     │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Install Mem0
pip install mem0ai

# Install our framework dependencies
pip install -r requirements.txt

# Optional: Vector database (if using Mem0's hosted service, skip this)
# For Qdrant:
pip install qdrant-client

# For Pinecone:
pip install pinecone-client
```

## Basic Usage

### 1. Initialize Both Systems

```python
from mem0 import Memory
from agentic_memory import AgenticMemory, MemoryScope
from examples.mem0_integration import Mem0AgenticBridge

# Initialize Mem0
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

# Initialize our framework
agentic = AgenticMemory()

# Create bridge
bridge = Mem0AgenticBridge(agentic, mem0_client=mem0)
```

### 2. Add Conversations (Mem0 Extracts)

```python
conversation = [
    {"role": "user", "content": "I'm building a chatbot using Python"},
    {"role": "assistant", "content": "Great! What framework?"},
    {"role": "user", "content": "I'm using FastAPI for the backend"},
    {"role": "assistant", "content": "Excellent choice!"},
]

# Mem0 automatically extracts facts and stores in our framework
bridge.add_conversation(
    messages=conversation,
    user_id="user_123",
    session_id="session_456"
)
```

### 3. Retrieve with Context Engineering

```python
# Our framework handles retrieval with advanced features
result = bridge.retrieve_with_context_engineering(
    query="What technology is the user using?",
    user_id="user_123",
    task_type="factual"
)

print(f"Retrieved {len(result.memories)} memories")
print(f"Strategy: {result.strategy_used}")
print(f"Problems solved: {result.metadata}")
```

## Memory Type Mapping

### Mem0 → Our Framework

| Mem0 Type | Our Type | Scope | Use Case |
|-----------|----------|-------|----------|
| User memory | Semantic | GLOBAL | User preferences, facts |
| Session memory | Episodic | SESSION | Conversation history |
| Agent memory | Procedural | PROJECT | Agent capabilities |

### Automatic Classification

The bridge automatically classifies memories:

```python
# Procedural indicators
"how to", "steps", "process", "workflow", "procedure"
→ Stored as Procedural memory

# Episodic indicators
"discussed", "mentioned", "said", "talked about"
→ Stored as Episodic memory

# Default
→ Stored as Semantic memory
```

## Advanced Features

### 1. Multi-User Isolation

```python
# Create isolated contexts
agentic.create_isolated_context("user_alice", MemoryScope.SESSION)
agentic.create_isolated_context("user_bob", MemoryScope.SESSION)

# Add conversations for each user
bridge.add_conversation(alice_messages, user_id="user_alice")
bridge.add_conversation(bob_messages, user_id="user_bob")

# Retrieve user-specific context (no mixing!)
alice_context = bridge.retrieve_with_context_engineering(
    query="What am I working on?",
    user_id="user_alice",
    scope=MemoryScope.SESSION
)
```

### 2. Hierarchical Scoping

```python
# Global: User preferences (accessible everywhere)
bridge.add_conversation(
    [{"role": "user", "content": "I prefer TypeScript"}],
    user_id="user_123"
)
# → Stored in GLOBAL scope

# Project: Project-specific info
agentic.store_semantic(
    entity="project_config",
    content="This project uses React 18",
    scope=MemoryScope.PROJECT
)

# Session: Current conversation
bridge.add_conversation(
    [{"role": "user", "content": "Let's fix the login bug"}],
    user_id="user_123",
    session_id="debug_session"
)
# → Stored in SESSION scope

# Retrieval respects hierarchy
result = agentic.retrieve(
    query="What should I know?",
    scope=MemoryScope.SESSION  # Gets SESSION + PROJECT + GLOBAL
)
```

### 3. Context Problem Solving

```python
# Add conflicting information
bridge.add_conversation([
    {"role": "user", "content": "I prefer verbose explanations"}
], user_id="user_123")

# Later...
bridge.add_conversation([
    {"role": "user", "content": "Actually, I prefer concise responses"}
], user_id="user_123")

# Our framework automatically resolves conflicts
result = bridge.retrieve_with_context_engineering(
    query="How does user prefer responses?",
    user_id="user_123"
)

# Returns most recent preference
print(result.metadata['conflicts_detected'])  # Shows conflict was detected
print(result.memories[0].content)  # "prefer concise responses"
```

### 4. Task-Based Retrieval

```python
# Factual queries → Semantic memory
result = bridge.retrieve_with_context_engineering(
    query="What is the user's preferred language?",
    user_id="user_123",
    task_type="factual"
)

# Procedural queries → Procedural memory
result = bridge.retrieve_with_context_engineering(
    query="How do I debug this issue?",
    user_id="user_123",
    task_type="procedural"
)

# Conversational queries → Episodic + Semantic
result = bridge.retrieve_with_context_engineering(
    query="What did we discuss last time?",
    user_id="user_123",
    task_type="conversational"
)
```

## Evaluation

Measure the effectiveness of the integrated system:

```python
from evaluation.evaluator import MemoryEvaluator, GoldenDataset

evaluator = MemoryEvaluator(agentic)

# Create test dataset
dataset = GoldenDataset()
dataset.add_query(
    query="What language does user prefer?",
    relevant_memory_ids=[mem1.id]
)

# Run evaluation
results = evaluator.run_full_evaluation(dataset, test_queries)

print(f"Precision: {results['retrieval']['precision']:.3f}")
print(f"Poisoning Rate: {results['context_problems']['poisoning_rate']:.3f}")
print(f"Latency p95: {results['performance']['latency_p95']:.2f} ms")
```

## Configuration Options

### Mem0 Configuration

```python
# Using Qdrant (local)
mem0_config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        }
    }
}

# Using Pinecone (cloud)
mem0_config = {
    "vector_store": {
        "provider": "pinecone",
        "config": {
            "api_key": "your-api-key",
            "environment": "us-west1-gcp"
        }
    }
}

# Using Mem0's hosted service
mem0_config = {
    "api_key": "your-mem0-api-key"
}
```

### Our Framework Configuration

```python
# Custom embedding model
agentic = AgenticMemory(
    embedding_model="all-mpnet-base-v2",  # More accurate
    enable_auto_pruning=True
)

# Custom scoring weights
from utils.scoring import RelevanceScorer

agentic.scorer = RelevanceScorer(
    semantic_weight=0.6,
    recency_weight=0.2,
    importance_weight=0.15,
    access_weight=0.05
)
```

## Best Practices

### 1. Use Mem0 for Extraction
Let Mem0 handle the heavy lifting of extracting facts from conversations.

### 2. Use Our Framework for Retrieval
Leverage our advanced retrieval features for better context quality.

### 3. Set Appropriate Scopes
- User preferences → GLOBAL
- Project config → PROJECT
- Conversation history → SESSION
- Temporary working memory → TASK

### 4. Monitor Performance
```python
# Check health regularly
is_healthy, alerts = evaluator.check_health()
if not is_healthy:
    print("Issues detected:", alerts)
```

### 5. Handle Conflicts
Enable conflict resolution to handle contradictory information:
```python
# Framework automatically detects and resolves
result = agentic.retrieve(query)
if result.metadata['conflicts_detected'] > 0:
    print("Conflicts resolved using recency strategy")
```

## Comparison: Before vs After Integration

### Before (Mem0 Only)

```python
# Simple but limited
mem0.add(messages, user_id="user_123")
results = mem0.search(query, user_id="user_123")

# Issues:
# ❌ No context problem solving
# ❌ No hierarchical scoping
# ❌ No isolation between contexts
# ❌ Basic retrieval only
```

### After (Integrated)

```python
# Powerful and comprehensive
bridge.add_conversation(messages, user_id="user_123")
result = bridge.retrieve_with_context_engineering(
    query=query,
    user_id="user_123",
    task_type="factual"
)

# Benefits:
# ✅ Automatic extraction (Mem0)
# ✅ Context problem solving (Our framework)
# ✅ Hierarchical scoping (Our framework)
# ✅ Multi-stage retrieval (Our framework)
# ✅ Evaluation metrics (Our framework)
```

## Example: Complete Chatbot

```python
class IntegratedChatbot:
    def __init__(self):
        self.mem0 = Memory.from_config(mem0_config)
        self.agentic = AgenticMemory()
        self.bridge = Mem0AgenticBridge(self.agentic, self.mem0)
    
    def chat(self, user_message: str, user_id: str, session_id: str):
        # 1. Retrieve relevant context
        context = self.bridge.retrieve_with_context_engineering(
            query=user_message,
            user_id=user_id,
            task_type="conversational"
        )
        
        # 2. Generate response with context
        response = self.generate_response(user_message, context)
        
        # 3. Store conversation (Mem0 extracts facts)
        self.bridge.add_conversation(
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response}
            ],
            user_id=user_id,
            session_id=session_id
        )
        
        return response
    
    def generate_response(self, message: str, context):
        # Use your LLM here with the optimized context
        context_str = "\n".join([m.content for m in context.memories])
        prompt = f"Context:\n{context_str}\n\nUser: {message}\nAssistant:"
        return llm.generate(prompt)

# Usage
bot = IntegratedChatbot()
response = bot.chat(
    "What technology should I use for my backend?",
    user_id="user_123",
    session_id="session_456"
)
```

## Running the Demo

```bash
# Run the integration demo
python examples/mem0_integration.py

# Expected output:
# - Basic integration demo
# - Multi-user isolation demo
# - Context problem solving demo
# - Performance comparison
```

## Troubleshooting

### Issue: Mem0 not extracting memories
**Solution**: Check your LLM configuration and API keys

### Issue: Context mixing between users
**Solution**: Use `create_isolated_context()` for each user

### Issue: High latency
**Solution**: Reduce `top_k` or adjust token budget

### Issue: Low retrieval quality
**Solution**: Run evaluation and adjust scoring weights

## Summary

**Integration Benefits:**
- 🚀 Automatic extraction (Mem0)
- 🎯 Advanced retrieval (Our framework)
- 🛡️ Context problem solving (Our framework)
- 📊 Measurable effectiveness (Our framework)
- 🔒 User isolation (Our framework)

**Best for:**
- Production chatbots
- Multi-user applications
- Complex agent systems
- Applications requiring high-quality context
