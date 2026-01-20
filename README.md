# Agentic Memory Framework

A production-ready framework for managing agent memory that solves critical context engineering problems:
- **Context Poisoning**: Incorrect/outdated information contaminating context
- **Context Distraction**: Irrelevant information diluting useful context
- **Context Confusion**: Ambiguous or conflicting information
- **Context Conflict**: Competing or contradictory memories

## Architecture

### Core Components

1. **Storage Layer** (Context Offloading)
   - Episodic, Semantic, and Procedural memory stores
   - External storage with metadata indexing

2. **Retrieval Orchestrator** (RAG + Tool Loadout)
   - Dynamic memory type selection
   - Multi-stage retrieval with reranking
   - Context budget management

3. **Context Refinement** (Pruning + Summarization)
   - Adaptive processing per memory type
   - Relevance-based pruning
   - Intelligent summarization

4. **Context Quarantine** (Isolation)
   - Scoped namespaces (Global → Project → Session → Task)
   - Boundary enforcement with controlled sharing

5. **Memory Management**
   - Conflict resolution
   - Quality validation
   - Feedback-based learning

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from agentic_memory import AgenticMemory

# Initialize framework
memory = AgenticMemory()

# Store memories
memory.store_episodic("user", "I prefer Python for backend development", scope="global")
memory.store_semantic("user_preference", {"language": "Python", "domain": "backend"})

# Retrieve with context
context = memory.retrieve("What language should I use?", task_type="recommendation")

# Use in your agent
response = agent.generate(query, context=context)
```

## Evaluation & Metrics

The framework includes comprehensive evaluation tools to measure memory effectiveness:

### Key Metrics
- **Retrieval Quality**: Precision, Recall, F1, MRR, NDCG
- **Context Problems**: Poisoning rate, distraction score, confusion events, clash resolution
- **Performance**: Latency (p50, p95, p99), token efficiency
- **Memory Management**: Utilization rate, freshness score, pruning accuracy

### Quick Evaluation
```python
from evaluation.evaluator import MemoryEvaluator, GoldenDataset

evaluator = MemoryEvaluator(memory)

# Create golden dataset
dataset = GoldenDataset()
dataset.add_query("What language for AI?", relevant_memory_ids=[mem1.id])

# Run evaluation
results = evaluator.run_full_evaluation(dataset, test_queries)

# Check health
is_healthy, alerts = evaluator.check_health()
```

See `EVALUATION.md` for detailed methodology and `examples/evaluation_demo.py` for demonstrations.

## Real-World Test Scenarios

See `tests/test_scenarios.py` for comprehensive test cases including:
- Customer support agent with conversation history
- Code assistant with project context
- Multi-project developer workflow
- Policy compliance checker
- Debugging assistant
- Memory evaluation and benchmarking

## Project Structure

```
agentic_memory/
├── core/
│   ├── storage.py          # Memory storage layer
│   ├── retrieval.py        # Retrieval orchestrator
│   ├── refinement.py       # Context refinement
│   └── quarantine.py       # Isolation management
├── models/
│   └── memory.py           # Memory data models
├── utils/
│   ├── embeddings.py       # Embedding utilities
│   └── scoring.py          # Relevance scoring
├── evaluation/
│   ├── metrics.py          # Evaluation metrics
│   └── evaluator.py        # Evaluation framework
├── tests/
│   ├── test_scenarios.py   # Real-world test cases
│   └── test_evaluation.py  # Evaluation tests
└── examples/
    ├── demo.py             # Usage demonstrations
    ├── quickstart.py       # Quick start guide
    └── evaluation_demo.py  # Evaluation examples
```
