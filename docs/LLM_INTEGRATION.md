# LLM Integration Guide

## Overview

This guide shows how to integrate the Agentic Memory Framework with Large Language Models (LLMs) to create memory-enhanced AI agents.

## Supported LLMs

- ✅ **OpenAI** (ChatGPT, GPT-4, GPT-3.5)
- ✅ **Google Gemini** (Gemini Pro)
- 🔄 **Anthropic Claude** (easily adaptable)
- 🔄 **Other LLMs** (any API-based LLM)

## Installation

```bash
# Install base framework
pip install -r requirements.txt

# Install LLM libraries
pip install openai                    # For OpenAI/ChatGPT
pip install google-generativeai       # For Google Gemini
pip install anthropic                 # For Claude (optional)
```

## Quick Start

### 1. Set API Keys

```bash
# Option 1: Environment variables (recommended)
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"

# Option 2: Set in code (for testing)
OPENAI_API_KEY = "your-key-here"
GOOGLE_API_KEY = "your-key-here"
```

### 2. Create Memory-Enhanced Agent

```python
from examples.llm_integration import MemoryEnhancedAgent, OpenAIClient
from agentic_memory import AgenticMemory

# Initialize
llm = OpenAIClient(api_key="your-key")
memory = AgenticMemory()
agent = MemoryEnhancedAgent(llm, memory, user_id="user_123")

# Chat with memory
response = agent.chat("I'm working on a Python project using FastAPI")
print(response)

# Store important facts
agent.remember_fact("User is working on a Python project using FastAPI")

# Later conversation - agent remembers!
response = agent.chat("Can you help me with authentication?")
# Agent will reference the FastAPI project from memory
```

### 3. Run Demos

```bash
python examples/llm_integration.py
```

## Architecture

```
User Input
    ↓
Memory Retrieval (relevant context)
    ↓
Context + User Input → LLM
    ↓
LLM Response
    ↓
Store in Memory
    ↓
Return to User
```

## Key Features

### 1. **Contextual Responses**

Without memory:
```
User: Can you help me with authentication?
Agent: Sure! What framework are you using?
```

With memory:
```
User: Can you help me with authentication?
Agent: For your FastAPI project, I recommend using JWT tokens...
```

### 2. **Multi-Session Continuity**

```python
# Session 1
agent1 = MemoryEnhancedAgent(llm, memory, user_id="alice")
agent1.chat("I prefer TypeScript")

# Session 2 (later)
agent2 = MemoryEnhancedAgent(llm, memory, user_id="alice")
agent2.chat("What language should I use?")
# Agent remembers: "Based on your preference for TypeScript..."
```

### 3. **Context Isolation**

```python
# Customer 1
agent_alice = MemoryEnhancedAgent(llm, memory, user_id="customer_alice")
agent_alice.chat("I have a login issue")

# Customer 2
agent_bob = MemoryEnhancedAgent(llm, memory, user_id="customer_bob")
agent_bob.chat("I have a payment issue")

# No context mixing - each customer's data is isolated!
```

### 4. **Preference Learning**

```python
agent.chat("That explanation was too technical")
agent.remember_preference("User prefers simple, non-technical explanations")

# Future responses adapt to this preference
agent.chat("What is machine learning?")
# Agent provides simpler explanation
```

## Demo Scenarios

### Demo 1: Basic Conversation (ChatGPT)
Shows how memory enhances basic chat interactions.

**Features:**
- Remembers user's project details
- Recalls preferences
- Provides contextual help

### Demo 2: Multi-Session (Gemini)
Demonstrates memory persistence across sessions.

**Features:**
- Session 1: User shares interests
- Session 2: Agent remembers from previous session
- Continuous context

### Demo 3: Code Assistant (ChatGPT)
Project-specific coding help with memory.

**Features:**
- Stores project configuration
- Remembers tech stack
- Provides relevant code examples

### Demo 4: Customer Support (Gemini)
Multi-customer support with isolation.

**Features:**
- Isolated contexts per customer
- No data mixing
- Personalized support

### Demo 5: Learning Preferences (ChatGPT)
Agent adapts based on user feedback.

**Features:**
- Learns communication style
- Adjusts complexity level
- Personalizes responses

### Demo 6: Conflict Resolution (Gemini)
Handles changing preferences.

**Features:**
- Detects conflicting information
- Uses most recent preference
- Automatic resolution

## Implementation Details

### Memory-Enhanced Agent Class

```python
class MemoryEnhancedAgent:
    def chat(self, user_message: str, task_type: str = "conversational"):
        # 1. Retrieve relevant memories
        memory_result = self.memory.retrieve(
            query=user_message,
            task_type=task_type,
            scope=MemoryScope.SESSION,
            context_id=self.user_id
        )
        
        # 2. Build context from memories
        context = self._build_context(memory_result.memories)
        
        # 3. Generate LLM response with context
        response = self.llm.generate(context + user_message)
        
        # 4. Store conversation in memory
        self._store_conversation(user_message, response)
        
        return response
```

### Context Building

```python
def _build_context(self, memories):
    context_parts = ["=== Relevant Context ==="]
    
    for mem in memories[:5]:  # Top 5 most relevant
        context_parts.append(f"[{mem.type.value}] {mem.content}")
    
    context_parts.append("=== End Context ===")
    return "\n".join(context_parts)
```

### Memory Storage

```python
def _store_conversation(self, user_msg, assistant_msg):
    # Store user message
    self.memory.store_episodic(
        source=self.user_id,
        content=f"User said: {user_msg}",
        scope=MemoryScope.SESSION
    )
    
    # Store assistant response
    self.memory.store_episodic(
        source="assistant",
        content=f"Assistant responded: {assistant_msg}",
        scope=MemoryScope.SESSION
    )
```

## Advanced Usage

### Custom Task Types

```python
# Factual queries → Uses semantic memory
response = agent.chat(
    "What is my preferred framework?",
    task_type="factual"
)

# Procedural queries → Uses procedural memory
response = agent.chat(
    "How do I deploy my app?",
    task_type="procedural"
)

# Recommendations → Uses semantic + procedural
response = agent.chat(
    "What should I use for authentication?",
    task_type="recommendation"
)
```

### Explicit Memory Management

```python
# Store facts explicitly
agent.remember_fact(
    "User's project uses microservices architecture",
    importance=0.9
)

# Store preferences
agent.remember_preference(
    "User prefers detailed code examples",
    importance=0.95
)

# Store procedural knowledge
memory.store_procedural(
    workflow_name="deployment",
    content="1) Run tests, 2) Build Docker image, 3) Deploy to K8s",
    scope=MemoryScope.PROJECT
)
```

### Memory Statistics

```python
stats = agent.get_memory_stats()
print(f"Total memories: {stats['total_memories']}")
print(f"By type: {stats['by_type']}")
print(f"By scope: {stats['by_scope']}")
```

## Best Practices

### 1. **Set Appropriate Scopes**

```python
# Global: User preferences (persist forever)
agent.remember_preference("User prefers Python", scope=MemoryScope.GLOBAL)

# Project: Project-specific info
memory.store_semantic("Tech stack: FastAPI + PostgreSQL", scope=MemoryScope.PROJECT)

# Session: Conversation history (temporary)
# Automatically stored by agent.chat()
```

### 2. **Use Task Types**

Always specify `task_type` for better memory selection:

```python
agent.chat("What is X?", task_type="factual")           # Facts
agent.chat("How do I Y?", task_type="procedural")       # Procedures
agent.chat("What should I use?", task_type="recommendation")  # Advice
agent.chat("Tell me about...", task_type="conversational")    # General
```

### 3. **Manage Token Budget**

```python
# Limit context size for faster responses
memory_result = memory.retrieve(
    query=user_message,
    max_tokens=1000  # Smaller context
)

# Or retrieve fewer memories
memory_result = memory.retrieve(
    query=user_message,
    top_k=3  # Only top 3 memories
)
```

### 4. **Handle API Errors**

```python
try:
    response = agent.chat(user_message)
except Exception as e:
    print(f"Error: {e}")
    response = "I'm having trouble right now. Please try again."
```

### 5. **Monitor Memory Growth**

```python
stats = agent.get_memory_stats()
if stats['total_memories'] > 1000:
    # Prune old, low-importance memories
    memory.prune_old_memories(days_threshold=30, importance_threshold=0.3)
```

## Comparison: With vs Without Memory

### Without Memory

```
User: I'm working on a FastAPI project
Agent: Great! How can I help?

[Later]
User: Can you help with authentication?
Agent: Sure! What framework are you using?  ← Forgot!
```

### With Memory

```
User: I'm working on a FastAPI project
Agent: Great! How can I help?
[Stores: "User working on FastAPI project"]

[Later]
User: Can you help with authentication?
Agent: For your FastAPI project, I recommend JWT tokens...  ← Remembers!
```

## Performance Considerations

### Latency Breakdown

```
Total Response Time: ~1-2 seconds
├── Memory Retrieval: 50-100ms
├── Context Building: 10-20ms
├── LLM API Call: 800-1500ms
└── Memory Storage: 20-50ms
```

### Optimization Tips

1. **Reduce top_k**: Fewer memories = faster retrieval
2. **Use task_type**: Better memory selection = less noise
3. **Limit max_tokens**: Smaller context = faster LLM processing
4. **Cache embeddings**: Reuse embeddings when possible

## Troubleshooting

### Issue: Agent doesn't remember

**Solution**: Check that facts are being stored
```python
# Explicitly store important information
agent.remember_fact("Important fact here")

# Verify storage
stats = agent.get_memory_stats()
print(stats)
```

### Issue: Irrelevant context retrieved

**Solution**: Use specific task types
```python
# Instead of:
agent.chat("How do I deploy?")

# Use:
agent.chat("How do I deploy?", task_type="procedural")
```

### Issue: High latency

**Solution**: Reduce context size
```python
memory_result = memory.retrieve(
    query=user_message,
    max_tokens=500,  # Smaller
    top_k=3          # Fewer memories
)
```

### Issue: API rate limits

**Solution**: Add retry logic
```python
import time

def chat_with_retry(agent, message, max_retries=3):
    for i in range(max_retries):
        try:
            return agent.chat(message)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                time.sleep(2 ** i)  # Exponential backoff
            else:
                raise
```

## Example: Complete Chatbot

```python
class ProductionChatbot:
    def __init__(self, llm_api_key: str):
        self.llm = OpenAIClient(llm_api_key)
        self.memory = AgenticMemory()
        self.agents = {}  # user_id -> agent
    
    def get_agent(self, user_id: str):
        if user_id not in self.agents:
            self.agents[user_id] = MemoryEnhancedAgent(
                self.llm, self.memory, user_id
            )
        return self.agents[user_id]
    
    def chat(self, user_id: str, message: str):
        agent = self.get_agent(user_id)
        return agent.chat(message)

# Usage
bot = ProductionChatbot(api_key="your-key")

# User 1
response = bot.chat("user_123", "I prefer Python")
# User 2
response = bot.chat("user_456", "I prefer JavaScript")

# Later - each user gets personalized responses
response = bot.chat("user_123", "What language should I use?")
# Returns: "Based on your preference, Python would be great..."
```

## Next Steps

1. **Run the demos**: `python examples/llm_integration.py`
2. **Try different LLMs**: Adapt the client wrappers
3. **Customize memory types**: Add domain-specific memory
4. **Evaluate performance**: Use the evaluation framework
5. **Deploy to production**: Add error handling, monitoring

## Summary

**Benefits of LLM + Memory Integration:**
- ✅ Contextual, personalized responses
- ✅ Multi-session continuity
- ✅ User preference learning
- ✅ Context isolation (multi-user)
- ✅ Automatic conflict resolution
- ✅ Improved user experience

The combination of LLMs and agentic memory creates truly intelligent, context-aware AI agents!
