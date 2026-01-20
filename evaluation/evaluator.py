import sys
sys.path.insert(0, '/Users/jchen65/dev/ai_playground/agentic_memory')

from typing import List, Dict, Any, Tuple, Optional
import time
from datetime import datetime
import numpy as np

from agentic_memory import AgenticMemory, MemoryType, MemoryScope
from evaluation.metrics import (
    RetrievalMetrics,
    ContextProblemMetrics,
    AgentPerformanceMetrics,
    MemoryManagementMetrics,
    MetricsAggregator
)


class GoldenDataset:
    """Labeled dataset for evaluation"""
    
    def __init__(self):
        self.queries = []
    
    def add_query(self, query: str, relevant_memory_ids: List[str], 
                  relevance_scores: Optional[Dict[str, float]] = None):
        self.queries.append({
            'query': query,
            'relevant_ids': set(relevant_memory_ids),
            'relevance_scores': relevance_scores or {mid: 1.0 for mid in relevant_memory_ids}
        })
    
    def __len__(self):
        return len(self.queries)
    
    def __iter__(self):
        return iter(self.queries)


class MemoryEvaluator:
    """Comprehensive memory system evaluator"""
    
    def __init__(self, memory_system: AgenticMemory):
        self.memory = memory_system
        self.metrics_aggregator = MetricsAggregator()
    
    def evaluate_retrieval_quality(self, golden_dataset: GoldenDataset) -> Dict[str, float]:
        precision_scores = []
        recall_scores = []
        f1_scores = []
        mrr_scores = []
        p_at_5_scores = []
        r_at_10_scores = []
        ndcg_at_10_scores = []
        
        for query_data in golden_dataset:
            query = query_data['query']
            relevant_ids = query_data['relevant_ids']
            relevance_scores = query_data['relevance_scores']
            
            result = self.memory.retrieve(query, top_k=10)
            retrieved_ids = [m.id for m in result.memories]
            retrieved_set = set(retrieved_ids)
            
            precision = RetrievalMetrics.precision(retrieved_set, relevant_ids)
            recall = RetrievalMetrics.recall(retrieved_set, relevant_ids)
            f1 = RetrievalMetrics.f1_score(precision, recall)
            mrr = RetrievalMetrics.mean_reciprocal_rank(retrieved_ids, relevant_ids)
            p_at_5 = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, 5)
            r_at_10 = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, 10)
            ndcg_at_10 = RetrievalMetrics.ndcg_at_k(retrieved_ids, relevance_scores, 10)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            mrr_scores.append(mrr)
            p_at_5_scores.append(p_at_5)
            r_at_10_scores.append(r_at_10)
            ndcg_at_10_scores.append(ndcg_at_10)
        
        results = {
            'precision': np.mean(precision_scores),
            'recall': np.mean(recall_scores),
            'f1': np.mean(f1_scores),
            'mrr': np.mean(mrr_scores),
            'precision@5': np.mean(p_at_5_scores),
            'recall@10': np.mean(r_at_10_scores),
            'ndcg@10': np.mean(ndcg_at_10_scores)
        }
        
        for metric, value in results.items():
            self.metrics_aggregator.record(f'retrieval_{metric}', value)
        
        return results
    
    def evaluate_context_problems(self, queries: List[str]) -> Dict[str, float]:
        poisoning_rates = []
        distraction_scores = []
        confusion_counts = []
        total_conflicts = 0
        resolved_conflicts = 0
        
        for query in queries:
            result = self.memory.retrieve(query)
            
            poisoning_rate = ContextProblemMetrics.poisoning_rate(result.memories)
            distraction_score = ContextProblemMetrics.distraction_score(result.memories, query)
            confusion_count = ContextProblemMetrics.confusion_events(result.memories)
            
            poisoning_rates.append(poisoning_rate)
            distraction_scores.append(distraction_score)
            confusion_counts.append(confusion_count)
            
            conflicts = result.metadata.get('conflicts_detected', 0)
            total_conflicts += conflicts
            resolved_conflicts += conflicts
        
        clash_resolution = ContextProblemMetrics.clash_resolution_rate(
            total_conflicts, resolved_conflicts
        )
        
        results = {
            'poisoning_rate': np.mean(poisoning_rates),
            'distraction_score': np.mean(distraction_scores),
            'confusion_events': np.sum(confusion_counts),
            'clash_resolution_rate': clash_resolution
        }
        
        for metric, value in results.items():
            self.metrics_aggregator.record(f'context_{metric}', value)
        
        return results
    
    def evaluate_performance(self, test_cases: List[Dict[str, Any]]) -> Dict[str, float]:
        latencies = []
        token_usages = []
        
        for case in test_cases:
            start_time = time.time()
            result = self.memory.retrieve(case['query'])
            latency = (time.time() - start_time) * 1000
            
            latencies.append(latency)
            token_usages.append(result.total_tokens)
        
        results = {
            'latency_mean': np.mean(latencies),
            'latency_p50': np.percentile(latencies, 50),
            'latency_p95': np.percentile(latencies, 95),
            'latency_p99': np.percentile(latencies, 99),
            'tokens_mean': np.mean(token_usages),
            'tokens_p95': np.percentile(token_usages, 95)
        }
        
        for metric, value in results.items():
            self.metrics_aggregator.record(f'performance_{metric}', value)
        
        return results
    
    def evaluate_memory_management(self) -> Dict[str, float]:
        all_memories = self.memory.store.get_all_memories()
        
        if not all_memories:
            return {
                'utilization_rate': 0.0,
                'avg_age_days': 0.0,
                'freshness_score': 0.0
            }
        
        accessed_memories = [m for m in all_memories if m.access_count > 0]
        utilization = MemoryManagementMetrics.utilization_rate(
            len(accessed_memories), len(all_memories)
        )
        
        avg_age = MemoryManagementMetrics.average_memory_age(all_memories)
        freshness = MemoryManagementMetrics.memory_freshness_score(all_memories)
        
        results = {
            'utilization_rate': utilization,
            'avg_age_days': avg_age,
            'freshness_score': freshness,
            'total_memories': len(all_memories),
            'accessed_memories': len(accessed_memories)
        }
        
        for metric, value in results.items():
            if metric not in ['total_memories', 'accessed_memories']:
                self.metrics_aggregator.record(f'management_{metric}', value)
        
        return results
    
    def run_full_evaluation(self, golden_dataset: GoldenDataset, 
                           test_queries: List[str]) -> Dict[str, Any]:
        print("Running full evaluation...")
        
        print("\n1. Evaluating retrieval quality...")
        retrieval_results = self.evaluate_retrieval_quality(golden_dataset)
        
        print("2. Evaluating context problems...")
        context_results = self.evaluate_context_problems(test_queries)
        
        print("3. Evaluating performance...")
        performance_results = self.evaluate_performance(
            [{'query': q} for q in test_queries]
        )
        
        print("4. Evaluating memory management...")
        management_results = self.evaluate_memory_management()
        
        return {
            'retrieval': retrieval_results,
            'context_problems': context_results,
            'performance': performance_results,
            'memory_management': management_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def compare_with_baseline(self, baseline_results: Dict[str, Any], 
                             current_results: Dict[str, Any]) -> Dict[str, float]:
        improvements = {}
        
        for category in ['retrieval', 'context_problems', 'performance']:
            if category in baseline_results and category in current_results:
                for metric, current_value in current_results[category].items():
                    if metric in baseline_results[category]:
                        baseline_value = baseline_results[category][metric]
                        
                        if baseline_value != 0:
                            improvement = ((current_value - baseline_value) / baseline_value) * 100
                            improvements[f'{category}_{metric}'] = improvement
        
        return improvements
    
    def get_metrics_summary(self, window_size: int = 100) -> Dict[str, Any]:
        return self.metrics_aggregator.get_all_summaries(window_size)
    
    def check_health(self) -> Tuple[bool, List[str]]:
        thresholds = {
            'retrieval_precision': ('<', 0.70),
            'retrieval_recall': ('<', 0.60),
            'retrieval_mrr': ('<', 0.60),
            'context_poisoning_rate': ('>', 0.05),
            'context_distraction_score': ('>', 0.20),
            'context_clash_resolution_rate': ('<', 0.90),
            'performance_latency_p95': ('>', 300),
            'management_utilization_rate': ('<', 0.25)
        }
        
        alerts = self.metrics_aggregator.check_thresholds(thresholds)
        is_healthy = len(alerts) == 0
        
        return is_healthy, alerts


class ABTestEvaluator:
    """Compare memory-enabled vs memory-disabled agents"""
    
    def __init__(self, memory_system: AgenticMemory):
        self.memory = memory_system
    
    def run_ab_test(self, test_cases: List[Dict[str, Any]], 
                   agent_with_memory, agent_without_memory) -> Dict[str, Any]:
        results_with = {'success': 0, 'latency': [], 'quality': []}
        results_without = {'success': 0, 'latency': [], 'quality': []}
        
        for case in test_cases:
            start = time.time()
            response_with = agent_with_memory(case['query'], self.memory)
            latency_with = time.time() - start
            
            results_with['latency'].append(latency_with)
            results_with['success'] += int(case.get('is_correct', lambda x: True)(response_with))
            results_with['quality'].append(case.get('rate_quality', lambda x: 3.0)(response_with))
            
            start = time.time()
            response_without = agent_without_memory(case['query'])
            latency_without = time.time() - start
            
            results_without['latency'].append(latency_without)
            results_without['success'] += int(case.get('is_correct', lambda x: True)(response_without))
            results_without['quality'].append(case.get('rate_quality', lambda x: 3.0)(response_without))
        
        n = len(test_cases)
        
        return {
            'success_rate_with_memory': results_with['success'] / n,
            'success_rate_without_memory': results_without['success'] / n,
            'success_rate_improvement': (results_with['success'] - results_without['success']) / n,
            'quality_with_memory': np.mean(results_with['quality']),
            'quality_without_memory': np.mean(results_without['quality']),
            'quality_improvement': np.mean(results_with['quality']) - np.mean(results_without['quality']),
            'latency_overhead': np.mean(results_with['latency']) - np.mean(results_without['latency'])
        }
