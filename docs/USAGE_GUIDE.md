# Agentic Memory Framework - Usage Guide

## Installation

```bash
cd /Users/jchen65/dev/ai_playground/agentic_memory
pip install -r requirements.txt
```

## Quick Start

```python
from agentic_memory import AgenticMemory, MemoryScope

# Initialize
memory = AgenticMemory()

# Store memories
memory.store_episodic(
    source="user",
    content="I'm building a web app with FastAPI",
    scope=MemoryScope.SESSION
)

# Retrieve relevant context
result = memory.retrieve(
    query="What framework am I using?",
    task_type="factual"
)

# Use in your agent
for mem in result.memories:
    print(f"[{mem.type.value}] {mem.content}")
```

## Core Concepts

### Memory Types

**Episodic Memory** - Specific events and conversations
```python
memory.store_episodic(
    source="user",
    content="Yesterday we discussed the authentication bug",
    scope=MemoryScope.SESSION,
    importance=0.7,
    tags=["bug", "authentication"]
)
```

**Semantic Memory** - Facts and preferences
```python
memory.store_semantic(
    entity="user_preference",
    content="User prefers Python over JavaScript",
    scope=MemoryScope.GLOBAL,
    importance=0.8,
    confidence=1.0
)
```

**Procedural Memory** - Workflows and patterns
```python
memory.store_procedural(
    workflow_name="code_review",
    content="Code review process: 1) Run tests, 2) Check style, 3) Review logic",
    scope=MemoryScope.PROJECT,
    importance=0.6
)
```

### Memory Scopes

**Hierarchy**: GLOBAL → PROJECT → SESSION → TASK

```python
# Global: Available everywhere
memory.store_semantic(
    entity="company_policy",
    content="All code must pass security review",
    scope=MemoryScope.GLOBAL
)

# Project: Available in project, session, and task
memory.store_semantic(
    entity="tech_stack",
    content="Project uses React and TypeScript",
    scope=MemoryScope.PROJECT
)

# Session: Available in session and task
memory.store_episodic(
    source="user",
    content="Today's goal: Fix the login bug",
    scope=MemoryScope.SESSION
)

# Task: Only available in current task
memory.store_episodic(
    source="system",
    content="Current step: Writing unit tests",
    scope=MemoryScope.TASK
)
```

## Retrieval Strategies

### Task-Based Retrieval

```python
# Factual queries - uses semantic memory
result = memory.retrieve(
    query="What is the user's preferred language?",
    task_type="factual"
)

# Recommendations - uses semantic + procedural
result = memory.retrieve(
    query="What framework should I use?",
    task_type="recommendation"
)

# Debugging - uses episodic + procedural
result = memory.retrieve(
    query="How did we fix the CORS issue?",
    task_type="debugging"
)

# Conversational - uses episodic + semantic
result = memory.retrieve(
    query="What did we discuss last time?",
    task_type="conversational"
)
```

### Scoped Retrieval

```python
# Retrieve only from specific scope
result = memory.retrieve(
    query="What are the project guidelines?",
    scope=MemoryScope.PROJECT
)

# Retrieve with context isolation
memory.create_isolated_context("customer_123", MemoryScope.SESSION)
result = memory.retrieve(
    query="What is the customer's issue?",
    context_id="customer_123",
    scope=MemoryScope.SESSION
)
```

### Advanced Retrieval

```python
from agentic_memory import MemoryType

result = memory.retrieve(
    query="Show me debugging procedures",
    memory_types=[MemoryType.PROCEDURAL],  # Only procedural memories
    max_tokens=1000,                        # Token budget
    top_k=5,                                # Max memories to return
    scope=MemoryScope.PROJECT
)

# Access metadata
print(f"Strategy used: {result.strategy_used}")
print(f"Total tokens: {result.total_tokens}")
print(f"Memory types: {result.metadata['memory_types_used']}")
print(f"Conflicts detected: {result.metadata['conflicts_detected']}")
```

## Context Isolation

### Multi-Customer Support

```python
# Create isolated contexts for each customer
memory.create_isolated_context("customer_alice", MemoryScope.SESSION)
memory.create_isolated_context("customer_bob", MemoryScope.SESSION)

# Store customer-specific data
memory.store_episodic(
    source="customer_alice",
    content="Alice has a login issue",
    scope=MemoryScope.SESSION,
    tags=["customer_alice"]
)

memory.store_episodic(
    source="customer_bob",
    content="Bob has a billing question",
    scope=MemoryScope.SESSION,
    tags=["customer_bob"]
)

# Retrieve only Alice's context
result = memory.retrieve(
    query="What is the issue?",
    context_id="customer_alice"
)
# Returns only Alice's memories
```

### Cross-Context Sharing

```python
# Grant tier 2 agent access to tier 1 context
memory.grant_context_access("tier2_agent", "tier1_agent")

# Tier 2 can now see tier 1's memories
result = memory.retrieve(
    query="What troubleshooting was done?",
    context_id="tier2_agent"
)
```

### Cleanup

```python
# Clear context when done
memory.clear_context("customer_alice")
```

## Memory Management

### Update Memory

```python
# Get memory ID when storing
mem = memory.store_semantic(
    entity="status",
    content="Project is in development",
    scope=MemoryScope.PROJECT
)

# Update later
memory.update_memory(
    mem.id,
    content="Project is in production",
    importance=0.9
)
```

### Delete Memory

```python
memory.delete_memory(mem.id)
```

### Pruning

```python
# Manual pruning
pruned_count = memory.prune_old_memories(
    days_threshold=30,        # Older than 30 days
    importance_threshold=0.2  # Less important than 0.2
)
print(f"Pruned {pruned_count} memories")

# Auto-pruning (enabled by default)
memory = AgenticMemory(enable_auto_pruning=True)
# Automatically prunes every 100 operations
```

### Statistics

```python
stats = memory.get_statistics()
print(stats)
# {
#   "total_memories": 42,
#   "by_type": {"episodic": 20, "semantic": 15, "procedural": 7},
#   "by_scope": {"global": 10, "project": 15, "session": 17},
#   "operation_count": 150
# }
```

### Export/Import

```python
# Export to file
memory.export_to_file("memories_backup.json")

# Import from file
new_memory = AgenticMemory()
new_memory.import_from_file("memories_backup.json")
```

## Real-World Examples

### Example 1: Code Assistant

```python
memory = AgenticMemory()

# Store project context
memory.store_semantic(
    entity="project_config",
    content="Project uses React 18, TypeScript, and Vite",
    scope=MemoryScope.PROJECT,
    tags=["tech_stack"]
)

# Store debugging history
memory.store_episodic(
    source="developer",
    content="Fixed CORS error by adding proxy in vite.config.ts",
    scope=MemoryScope.PROJECT,
    importance=0.8,
    tags=["debugging", "cors"]
)

# Store best practices
memory.store_procedural(
    workflow_name="component_creation",
    content="Create components in src/components with TypeScript and PropTypes",
    scope=MemoryScope.PROJECT
)

# Retrieve when coding
result = memory.retrieve(
    query="How should I create a new component?",
    task_type="procedural"
)
```

### Example 2: Customer Support

```python
memory = AgenticMemory()

# Create isolated context per customer
memory.create_isolated_context("ticket_12345", MemoryScope.SESSION)

# Store conversation
memory.store_episodic(
    source="customer",
    content="Customer reports slow page load times on mobile",
    scope=MemoryScope.SESSION,
    tags=["ticket_12345", "performance"]
)

memory.store_episodic(
    source="agent",
    content="Suggested clearing cache and checking network speed",
    scope=MemoryScope.SESSION,
    tags=["ticket_12345", "troubleshooting"]
)

# Retrieve full context
result = memory.retrieve(
    query="What troubleshooting steps were taken?",
    context_id="ticket_12345",
    task_type="conversational"
)
```

### Example 3: Multi-Project Developer

```python
memory = AgenticMemory()

# Store global preferences
memory.store_semantic(
    entity="user_preference",
    content="Developer prefers TypeScript and functional programming",
    scope=MemoryScope.GLOBAL
)

# Store project-specific conventions
memory.store_semantic(
    entity="project_alpha",
    content="Project Alpha uses snake_case naming",
    scope=MemoryScope.PROJECT,
    tags=["project_alpha", "conventions"]
)

memory.store_semantic(
    entity="project_beta",
    content="Project Beta uses camelCase naming",
    scope=MemoryScope.PROJECT,
    tags=["project_beta", "conventions"]
)

# Retrieve project-specific context
result = memory.retrieve(
    query="What naming convention should I use?",
    scope=MemoryScope.PROJECT
)
# Framework handles conflicts automatically
```

## Best Practices

### 1. Choose Appropriate Scopes

- **GLOBAL**: User preferences, company policies, universal facts
- **PROJECT**: Project-specific config, tech stack, conventions
- **SESSION**: Conversation history, current work context
- **TASK**: Temporary working memory, current subtask

### 2. Set Importance Correctly

- **1.0**: Critical (security policies, breaking changes)
- **0.7-0.9**: Important (user preferences, key decisions)
- **0.4-0.6**: Normal (regular conversations, standard info)
- **0.1-0.3**: Low (routine logs, temporary notes)

### 3. Use Tags for Organization

```python
memory.store_episodic(
    source="user",
    content="Fixed authentication bug in login endpoint",
    scope=MemoryScope.PROJECT,
    tags=["bug_fix", "authentication", "security", "login"]
)
```

### 4. Leverage Task Types

Always specify `task_type` for better memory selection:
- `"factual"` - Looking up facts
- `"recommendation"` - Getting suggestions
- `"debugging"` - Troubleshooting issues
- `"procedural"` - Learning how to do something
- `"conversational"` - Maintaining dialogue context

### 5. Handle Conflicts

```python
# Store with confidence scores
memory.store_semantic(
    entity="api_endpoint",
    content="API endpoint is /api/v1/users",
    confidence=0.8  # Not 100% sure
)

# Later, update with higher confidence
memory.store_semantic(
    entity="api_endpoint",
    content="API endpoint is /api/v2/users",
    confidence=1.0  # Confirmed
)
# Framework will prefer higher confidence in conflicts
```

### 6. Monitor and Prune

```python
# Check statistics regularly
stats = memory.get_statistics()
if stats["total_memories"] > 1000:
    memory.prune_old_memories(days_threshold=60)
```

## Troubleshooting

### No Memories Retrieved

```python
result = memory.retrieve(query="...", scope=MemoryScope.PROJECT)
if not result.memories:
    # Check if memories exist in that scope
    all_memories = memory.store.get_all_memories(scope=MemoryScope.PROJECT)
    print(f"Total memories in PROJECT scope: {len(all_memories)}")
```

### Too Many Irrelevant Results

```python
# Increase min_relevance threshold
result = memory.retrieve(
    query="...",
    min_relevance=0.5  # Default is 0.3
)

# Or specify exact memory types
result = memory.retrieve(
    query="...",
    memory_types=[MemoryType.SEMANTIC]
)
```

### Context Too Large

```python
# Reduce max_tokens
result = memory.retrieve(
    query="...",
    max_tokens=1000  # Default is 2000
)

# Or reduce top_k
result = memory.retrieve(
    query="...",
    top_k=3  # Default is 10
)
```

## Running Examples

```bash
# Quick start
python examples/quickstart.py

# Comprehensive demos
python examples/demo.py

# Run test scenarios
pytest tests/test_scenarios.py -v

# Run specific test class
pytest tests/test_scenarios.py::TestCustomerSupportAgent -v
```

## Next Steps

1. Read `ARCHITECTURE.md` for deep dive into system design
2. Explore `tests/test_scenarios.py` for real-world use cases
3. Run `examples/demo.py` to see all features in action
4. Integrate into your agent system
5. Customize scoring, refinement, or embedding models as needed
