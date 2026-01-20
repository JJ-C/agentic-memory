# Agentic Memory Framework - Architecture

## Overview

This framework addresses four critical context engineering problems:
1. **Context Poisoning**: Incorrect/outdated information contaminating context
2. **Context Distraction**: Irrelevant information diluting useful context
3. **Context Confusion**: Ambiguous or conflicting information
4. **Context Clash**: Competing or contradictory memories

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AgenticMemory API                        │
│  (Main interface for storing and retrieving memories)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Retrieval Orchestrator                      │
│  • Dynamic memory type selection (Tool Loadout)             │
│  • Multi-stage retrieval pipeline                           │
│  • Conflict detection and resolution                        │
└─────────────────────────────────────────────────────────────┘
           │              │              │              │
           ▼              ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Storage  │  │Quarantine│  │Refinement│  │ Scoring  │
    │  Layer   │  │  Layer   │  │  Layer   │  │  Layer   │
    └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

## Core Components

### 1. Storage Layer (Context Offloading)

**File**: `core/storage.py`

**Purpose**: External memory store with efficient indexing

**Features**:
- Multiple memory types (Episodic, Semantic, Procedural)
- Vector embeddings for semantic search
- Multi-index support (scope, type, tags)
- Import/export for persistence

**Addresses**:
- ✅ Context Offloading: Store unlimited memories externally
- ✅ Context Poisoning: Track metadata and confidence scores

### 2. Retrieval Orchestrator (RAG + Tool Loadout)

**File**: `core/retrieval.py`

**Purpose**: Intelligent memory selection and retrieval

**Pipeline**:
```
Query → Intent Classification → Memory Type Selection → 
Semantic Search → Quarantine Filter → Reranking → 
Conflict Resolution → Refinement → Result
```

**Features**:
- Dynamic memory type selection based on task
- Multi-stage retrieval (coarse → fine)
- Relevance scoring with multiple factors
- Conflict detection and resolution

**Addresses**:
- ✅ Tool Loadout: Select only relevant memory types
- ✅ Context Distraction: Filter by relevance
- ✅ Context Clash: Resolve contradictions

### 3. Context Quarantine (Isolation)

**File**: `core/quarantine.py`

**Purpose**: Scope-based memory isolation

**Hierarchy**:
```
GLOBAL (accessible from all scopes)
  ↓
PROJECT (accessible from project, session, task)
  ↓
SESSION (accessible from session, task)
  ↓
TASK (only accessible from task)
```

**Features**:
- Hierarchical scope access control
- Cross-boundary permissions
- Context isolation for multi-agent systems

**Addresses**:
- ✅ Context Confusion: Clear scope boundaries
- ✅ Context Clash: Prevent inappropriate mixing

### 4. Context Refinement (Pruning + Summarization)

**File**: `core/refinement.py`

**Purpose**: Optimize retrieved context for LLM consumption

**Strategies**:
- **Episodic memories**: Summarize long conversations
- **Semantic memories**: Prune irrelevant sentences
- **Token budget**: Fit within context limits

**Features**:
- Adaptive processing per memory type
- Query-aware pruning
- Hierarchical summarization

**Addresses**:
- ✅ Context Distraction: Remove irrelevant details
- ✅ Token efficiency: Maximize information density

### 5. Relevance Scoring

**File**: `utils/scoring.py`

**Purpose**: Multi-factor relevance ranking

**Scoring Formula**:
```
score = (semantic_weight × semantic_similarity +
         recency_weight × recency_score +
         importance_weight × importance +
         access_weight × access_score) × confidence
```

**Factors**:
- **Semantic similarity**: Vector cosine distance
- **Recency**: Exponential decay over time
- **Importance**: User-defined or learned
- **Access frequency**: Usage-based boosting
- **Confidence**: Reliability score

**Addresses**:
- ✅ Context Distraction: Prioritize relevant memories
- ✅ Context Poisoning: Weight by confidence

## Memory Types

### Episodic Memory
- **What**: Specific events and conversations
- **When**: Temporal sequences, past interactions
- **Use cases**: Conversation history, debugging logs
- **Scope**: Typically SESSION or TASK

### Semantic Memory
- **What**: Facts, preferences, relationships
- **When**: Timeless knowledge
- **Use cases**: User preferences, project configuration
- **Scope**: Typically GLOBAL or PROJECT

### Procedural Memory
- **What**: Workflows, patterns, how-to knowledge
- **When**: Repeatable processes
- **Use cases**: Best practices, debugging procedures
- **Scope**: Typically PROJECT

## Retrieval Strategies

### Task-Based Memory Selection

| Task Type | Memory Types Used | Rationale |
|-----------|------------------|-----------|
| `factual` | Semantic | Direct fact lookup |
| `recommendation` | Semantic, Procedural | Preferences + patterns |
| `conversational` | Episodic, Semantic | History + facts |
| `procedural` | Procedural, Episodic | Workflows + examples |
| `debugging` | Episodic, Procedural | Past issues + solutions |
| `general` | All types | Comprehensive search |

### Multi-Stage Retrieval

1. **Stage 1: Coarse Retrieval**
   - Semantic search with top_k × 5
   - Fast vector similarity
   - Cast wide net

2. **Stage 2: Quarantine Filter**
   - Apply scope restrictions
   - Enforce access control
   - Remove out-of-scope memories

3. **Stage 3: Reranking**
   - Multi-factor scoring
   - Consider recency, importance, access
   - Precision ranking

4. **Stage 4: Conflict Resolution**
   - Detect contradictions
   - Apply resolution strategy (recency/confidence/importance)
   - Remove losers

5. **Stage 5: Refinement**
   - Prune irrelevant content
   - Summarize long memories
   - Fit to token budget

## Conflict Resolution Strategies

### Detection
- Keyword-based contradiction detection
- Negation pairs (prefer/dislike, enable/disable)
- Same entity, different values

### Resolution Policies
1. **Recency**: Prefer newer information (default)
2. **Confidence**: Prefer higher confidence scores
3. **Importance**: Prefer more important memories
4. **User override**: Explicit user selection

## Memory Management

### Importance Decay
```python
importance_new = max(0.0, importance - (decay_rate × days_elapsed))
```

### Access-Based Boosting
```python
frequency_score = log(access_count + 1) / 5
recency_factor = exp(-0.05 × hours_since_access)
access_score = frequency_score × 0.7 + recency_factor × 0.3
```

### Auto-Pruning
- Triggered every 100 operations
- Removes memories older than threshold
- Removes memories below importance threshold
- Configurable per instance

## Performance Considerations

### Embedding Model
- Default: `all-MiniLM-L6-v2` (fast, 384 dimensions)
- Alternatives: `all-mpnet-base-v2` (slower, more accurate)
- Trade-off: Speed vs. accuracy

### Indexing
- In-memory indexes for fast lookup
- Scope, type, and tag indexes
- O(1) access by ID
- O(log n) sorted retrieval

### Scalability
- Current: In-memory storage (prototype)
- Production: Replace with vector DB (Pinecone, Weaviate, ChromaDB)
- Horizontal scaling: Shard by scope or user

## Integration Patterns

### Pattern 1: Single Agent
```python
memory = AgenticMemory()
context = memory.retrieve(query, task_type="recommendation")
response = agent.generate(query, context=context)
```

### Pattern 2: Multi-Agent with Isolation
```python
memory = AgenticMemory()

# Create isolated contexts
memory.create_isolated_context("agent_a", MemoryScope.SESSION)
memory.create_isolated_context("agent_b", MemoryScope.SESSION)

# Each agent retrieves only its context
context_a = memory.retrieve(query, context_id="agent_a")
context_b = memory.retrieve(query, context_id="agent_b")
```

### Pattern 3: Hierarchical Agents
```python
# Coordinator has global scope
coordinator_context = memory.retrieve(query, scope=MemoryScope.GLOBAL)

# Specialists have project scope
specialist_context = memory.retrieve(query, scope=MemoryScope.PROJECT)
```

## Extension Points

### Custom Embedding Models
```python
from utils.embeddings import EmbeddingManager

custom_embeddings = EmbeddingManager("your-model-name")
memory = AgenticMemory()
memory.embeddings = custom_embeddings
```

### Custom Scoring
```python
from utils.scoring import RelevanceScorer

custom_scorer = RelevanceScorer(
    semantic_weight=0.6,
    recency_weight=0.3,
    importance_weight=0.1
)
memory.scorer = custom_scorer
```

### Custom Refinement
```python
from core.refinement import ContextRefinement

custom_refinement = ContextRefinement(max_summary_ratio=0.2)
memory.refinement = custom_refinement
```

## Future Enhancements

1. **Graph-based memory**: Add relationship modeling
2. **Memory consolidation**: Automatic merging and summarization
3. **Reflection layer**: Agent self-analysis of memory quality
4. **Multi-modal memories**: Support images, code, structured data
5. **Distributed storage**: Production-grade vector database integration
6. **Memory versioning**: Track evolution of facts over time
7. **Privacy controls**: Fine-grained access policies
8. **Audit logging**: Track memory access for compliance
