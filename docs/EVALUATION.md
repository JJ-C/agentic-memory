# Evaluating Agent Memory Effectiveness

## Overview

Measuring memory effectiveness requires evaluating both **retrieval quality** and **impact on agent performance**. This guide provides metrics, benchmarks, and evaluation methodologies.

## Core Evaluation Dimensions

### 1. Retrieval Quality Metrics

#### Precision & Recall
```python
# Precision: Of retrieved memories, how many are relevant?
precision = relevant_retrieved / total_retrieved

# Recall: Of all relevant memories, how many were retrieved?
recall = relevant_retrieved / total_relevant

# F1 Score: Harmonic mean
f1 = 2 * (precision * recall) / (precision + recall)
```

#### Mean Reciprocal Rank (MRR)
```python
# Position of first relevant result
# Higher is better (1.0 = first result is relevant)
mrr = 1 / rank_of_first_relevant_result
```

#### Normalized Discounted Cumulative Gain (NDCG)
```python
# Measures ranking quality with graded relevance
# Accounts for position - earlier results weighted more
ndcg = dcg / ideal_dcg
```

### 2. Context Engineering Problem Metrics

#### Context Poisoning Rate
```python
# Percentage of retrieved memories that are outdated/incorrect
poisoning_rate = incorrect_memories / total_retrieved

# Target: < 5%
```

#### Context Distraction Score
```python
# Percentage of retrieved content that's irrelevant
distraction_score = irrelevant_tokens / total_tokens

# Target: < 20%
```

#### Context Confusion Events
```python
# Number of ambiguous/unclear retrievals
confusion_events = count_ambiguous_results

# Target: 0 per 100 queries
```

#### Context Clash Resolution Rate
```python
# Percentage of conflicts successfully resolved
resolution_rate = conflicts_resolved / conflicts_detected

# Target: > 95%
```

### 3. Agent Performance Metrics

#### Task Success Rate
```python
# With vs without memory
success_with_memory = completed_tasks / total_tasks_with_memory
success_without_memory = completed_tasks / total_tasks_without_memory

memory_impact = success_with_memory - success_without_memory
```

#### Response Quality
```python
# Human evaluation or LLM-as-judge
quality_score = rate_response(relevance, accuracy, completeness)

# Scale: 1-5
# Target: > 4.0 with memory
```

#### Token Efficiency
```python
# Information density
efficiency = useful_information / tokens_used

# Target: > 0.7 (70% of tokens are useful)
```

#### Latency
```python
# Retrieval overhead
retrieval_latency = time_to_retrieve_and_process

# Target: < 200ms for most queries
```

### 4. Memory Management Metrics

#### Memory Utilization
```python
# Percentage of memories that get accessed
utilization_rate = accessed_memories / total_memories

# Target: > 30% (avoid storing unused memories)
```

#### Pruning Accuracy
```python
# Of pruned memories, how many were correctly identified as low-value?
pruning_accuracy = correctly_pruned / total_pruned

# Target: > 90%
```

#### Memory Freshness
```python
# Average age of retrieved memories
avg_age = sum(current_time - memory.timestamp) / count

# Context-dependent, but track trends
```

## Evaluation Methodologies

### Method 1: Golden Dataset Evaluation

Create labeled test sets with known relevant memories:

```python
class MemoryEvaluator:
    def __init__(self, memory_system, golden_dataset):
        self.memory = memory_system
        self.dataset = golden_dataset
    
    def evaluate_retrieval(self):
        results = {
            'precision': [],
            'recall': [],
            'mrr': [],
            'ndcg': []
        }
        
        for query, relevant_ids in self.dataset:
            retrieved = self.memory.retrieve(query)
            retrieved_ids = {m.id for m in retrieved.memories}
            relevant_set = set(relevant_ids)
            
            # Precision
            if retrieved_ids:
                precision = len(retrieved_ids & relevant_set) / len(retrieved_ids)
                results['precision'].append(precision)
            
            # Recall
            if relevant_set:
                recall = len(retrieved_ids & relevant_set) / len(relevant_set)
                results['recall'].append(recall)
            
            # MRR
            for i, mem in enumerate(retrieved.memories, 1):
                if mem.id in relevant_set:
                    results['mrr'].append(1.0 / i)
                    break
            else:
                results['mrr'].append(0.0)
        
        return {
            'precision': np.mean(results['precision']),
            'recall': np.mean(results['recall']),
            'mrr': np.mean(results['mrr']),
            'f1': self._calculate_f1(results['precision'], results['recall'])
        }
    
    def _calculate_f1(self, precision_list, recall_list):
        f1_scores = []
        for p, r in zip(precision_list, recall_list):
            if p + r > 0:
                f1_scores.append(2 * p * r / (p + r))
        return np.mean(f1_scores) if f1_scores else 0.0
```

### Method 2: A/B Testing

Compare memory-enabled vs memory-disabled agents:

```python
def ab_test_memory_impact(test_cases, num_trials=100):
    results = {
        'with_memory': {'success': 0, 'quality': [], 'latency': []},
        'without_memory': {'success': 0, 'quality': [], 'latency': []}
    }
    
    for case in test_cases:
        # With memory
        start = time.time()
        response_with = agent_with_memory(case.query)
        latency_with = time.time() - start
        
        results['with_memory']['success'] += int(case.is_correct(response_with))
        results['with_memory']['quality'].append(case.rate_quality(response_with))
        results['with_memory']['latency'].append(latency_with)
        
        # Without memory
        start = time.time()
        response_without = agent_without_memory(case.query)
        latency_without = time.time() - start
        
        results['without_memory']['success'] += int(case.is_correct(response_without))
        results['without_memory']['quality'].append(case.rate_quality(response_without))
        results['without_memory']['latency'].append(latency_without)
    
    return {
        'success_rate_delta': (
            results['with_memory']['success'] - results['without_memory']['success']
        ) / len(test_cases),
        'quality_delta': (
            np.mean(results['with_memory']['quality']) - 
            np.mean(results['without_memory']['quality'])
        ),
        'latency_overhead': (
            np.mean(results['with_memory']['latency']) - 
            np.mean(results['without_memory']['latency'])
        )
    }
```

### Method 3: LLM-as-Judge Evaluation

Use an LLM to evaluate retrieval quality:

```python
def llm_judge_evaluation(query, retrieved_memories, ground_truth_context):
    prompt = f"""
    Evaluate the quality of retrieved memories for this query.
    
    Query: {query}
    
    Retrieved Memories:
    {format_memories(retrieved_memories)}
    
    Ground Truth Context:
    {ground_truth_context}
    
    Rate on a scale of 1-5:
    1. Relevance: How relevant are the retrieved memories?
    2. Completeness: Is all necessary information present?
    3. Accuracy: Is the information correct and up-to-date?
    4. Conciseness: Is there minimal irrelevant information?
    
    Provide scores and brief justification.
    """
    
    response = llm.evaluate(prompt)
    return parse_scores(response)
```

### Method 4: User Feedback Loop

Collect implicit and explicit feedback:

```python
class FeedbackCollector:
    def track_implicit_feedback(self, memory_id, action):
        """Track user actions as implicit feedback"""
        feedback_signals = {
            'used_in_response': +0.1,      # Memory was helpful
            'ignored': -0.05,               # Memory wasn't used
            'corrected': -0.2,              # User corrected information
            'repeated_query': -0.1          # User had to ask again
        }
        
        if action in feedback_signals:
            memory = self.memory.get_memory(memory_id)
            memory.boost_importance(feedback_signals[action])
    
    def collect_explicit_feedback(self, memory_id, rating):
        """Direct user rating: thumbs up/down"""
        memory = self.memory.get_memory(memory_id)
        
        if rating == 'positive':
            memory.boost_importance(0.2)
            memory.confidence = min(1.0, memory.confidence + 0.1)
        elif rating == 'negative':
            memory.importance = max(0.0, memory.importance - 0.3)
            memory.confidence = max(0.0, memory.confidence - 0.2)
```

## Benchmark Datasets

### Synthetic Benchmarks

```python
def create_synthetic_benchmark():
    """Create controlled test scenarios"""
    benchmark = []
    
    # Test 1: Recency preference
    benchmark.append({
        'name': 'recency_test',
        'memories': [
            ('Old info: API is v1', timestamp='2023-01-01', relevant=False),
            ('New info: API is v2', timestamp='2024-01-01', relevant=True)
        ],
        'query': 'What API version should I use?',
        'expected_top': 'New info: API is v2'
    })
    
    # Test 2: Scope isolation
    benchmark.append({
        'name': 'scope_test',
        'memories': [
            ('Project A uses React', scope='project_a', relevant=False),
            ('Project B uses Vue', scope='project_b', relevant=True)
        ],
        'query': 'What framework does project B use?',
        'context': 'project_b',
        'expected_top': 'Project B uses Vue'
    })
    
    # Test 3: Conflict resolution
    benchmark.append({
        'name': 'conflict_test',
        'memories': [
            ('User prefers verbose', timestamp='2024-01-01', relevant=False),
            ('User prefers concise', timestamp='2024-06-01', relevant=True)
        ],
        'query': 'How does user prefer responses?',
        'expected_top': 'User prefers concise'
    })
    
    return benchmark
```

### Real-World Benchmarks

Use existing datasets:
- **MS MARCO**: Question answering
- **Natural Questions**: Information retrieval
- **SQuAD**: Reading comprehension
- **ConvAI2**: Conversational memory

Adapt to memory framework format:
```python
def adapt_marco_to_memory_format(marco_dataset):
    memories = []
    queries = []
    
    for item in marco_dataset:
        # Store passages as memories
        for passage in item['passages']:
            memories.append({
                'content': passage['text'],
                'type': MemoryType.SEMANTIC,
                'scope': MemoryScope.GLOBAL,
                'relevance_label': passage['is_selected']
            })
        
        queries.append({
            'query': item['query'],
            'relevant_ids': [p['id'] for p in item['passages'] if p['is_selected']]
        })
    
    return memories, queries
```

## Continuous Monitoring

### Production Metrics Dashboard

Track these metrics in production:

```python
class MemoryMetrics:
    def __init__(self):
        self.metrics = {
            'retrieval_latency_p50': [],
            'retrieval_latency_p95': [],
            'retrieval_latency_p99': [],
            'memories_retrieved_avg': [],
            'token_usage_avg': [],
            'cache_hit_rate': [],
            'conflict_detection_rate': [],
            'pruning_rate': []
        }
    
    def record_retrieval(self, result, latency):
        self.metrics['retrieval_latency_p50'].append(latency)
        self.metrics['memories_retrieved_avg'].append(len(result.memories))
        self.metrics['token_usage_avg'].append(result.total_tokens)
        
        if result.metadata.get('conflicts_detected', 0) > 0:
            self.metrics['conflict_detection_rate'].append(1)
        else:
            self.metrics['conflict_detection_rate'].append(0)
    
    def get_summary(self, window='1h'):
        return {
            'latency_p50': np.percentile(self.metrics['retrieval_latency_p50'], 50),
            'latency_p95': np.percentile(self.metrics['retrieval_latency_p50'], 95),
            'latency_p99': np.percentile(self.metrics['retrieval_latency_p50'], 99),
            'avg_memories': np.mean(self.metrics['memories_retrieved_avg']),
            'avg_tokens': np.mean(self.metrics['token_usage_avg']),
            'conflict_rate': np.mean(self.metrics['conflict_detection_rate'])
        }
```

### Alerting Thresholds

```python
ALERT_THRESHOLDS = {
    'retrieval_latency_p95': 500,      # ms
    'poisoning_rate': 0.05,             # 5%
    'distraction_score': 0.30,          # 30%
    'conflict_resolution_rate': 0.90,   # 90%
    'memory_utilization': 0.20,         # 20%
    'pruning_accuracy': 0.85            # 85%
}

def check_alerts(current_metrics):
    alerts = []
    
    for metric, threshold in ALERT_THRESHOLDS.items():
        if metric.endswith('_rate') and current_metrics[metric] < threshold:
            alerts.append(f"⚠️ {metric} below threshold: {current_metrics[metric]:.2%}")
        elif current_metrics[metric] > threshold:
            alerts.append(f"⚠️ {metric} above threshold: {current_metrics[metric]}")
    
    return alerts
```

## Evaluation Best Practices

### 1. Multi-Dimensional Evaluation
Don't rely on a single metric. Evaluate:
- Retrieval quality (precision, recall)
- Context problem rates (poisoning, distraction, confusion, clash)
- Agent performance (task success, response quality)
- System performance (latency, token usage)

### 2. Domain-Specific Benchmarks
Create benchmarks specific to your use case:
- Customer support: Ticket resolution accuracy
- Code assistant: Code suggestion relevance
- Conversational: Coherence and context maintenance

### 3. Temporal Evaluation
Test memory performance over time:
- Does quality degrade as memories accumulate?
- Is pruning effective?
- Are conflicts increasing?

### 4. Edge Case Testing
Specifically test:
- Empty memory state
- Conflicting information
- Ambiguous queries
- Out-of-scope queries
- Very long conversations

### 5. Human-in-the-Loop
Combine automated metrics with human evaluation:
- Sample random retrievals for manual review
- Collect user feedback
- Conduct user studies

## Target Benchmarks

Based on research and best practices:

| Metric | Target | Excellent |
|--------|--------|-----------|
| Precision@5 | > 0.70 | > 0.85 |
| Recall@10 | > 0.60 | > 0.80 |
| MRR | > 0.60 | > 0.80 |
| Context Poisoning | < 5% | < 2% |
| Context Distraction | < 20% | < 10% |
| Conflict Resolution | > 90% | > 95% |
| Task Success Improvement | > 15% | > 30% |
| Retrieval Latency (p95) | < 300ms | < 150ms |
| Memory Utilization | > 25% | > 40% |

## Example Evaluation Script

See `tests/test_evaluation.py` for a complete implementation that includes:
- Golden dataset creation
- Automated metric calculation
- A/B testing framework
- LLM-as-judge integration
- Production monitoring setup

## Continuous Improvement Loop

```
1. Collect Metrics → 2. Identify Issues → 3. Adjust Parameters → 4. Re-evaluate
     ↑                                                                    ↓
     └────────────────────────────────────────────────────────────────────┘
```

**Example adjustments based on metrics:**
- Low precision → Increase `min_relevance` threshold
- High distraction → Increase pruning aggressiveness
- High latency → Reduce `top_k` or optimize embeddings
- Low utilization → Adjust importance scoring weights
- High conflicts → Improve conflict detection rules

## Summary

Effective memory evaluation requires:
1. **Multiple metrics** across retrieval, context problems, and agent performance
2. **Continuous monitoring** in production with alerting
3. **Regular benchmarking** against golden datasets
4. **User feedback** integration for real-world validation
5. **Iterative improvement** based on measured results

The goal is not perfect retrieval, but **measurably better agent performance** with **acceptable overhead**.
