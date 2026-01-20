import numpy as np
from typing import List, Set, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict


class RetrievalMetrics:
    """Calculate retrieval quality metrics"""
    
    @staticmethod
    def precision(retrieved_ids: Set[str], relevant_ids: Set[str]) -> float:
        if not retrieved_ids:
            return 0.0
        return len(retrieved_ids & relevant_ids) / len(retrieved_ids)
    
    @staticmethod
    def recall(retrieved_ids: Set[str], relevant_ids: Set[str]) -> float:
        if not relevant_ids:
            return 0.0
        return len(retrieved_ids & relevant_ids) / len(relevant_ids)
    
    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        for i, rid in enumerate(retrieved_ids, 1):
            if rid in relevant_ids:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        if k == 0:
            return 0.0
        retrieved_at_k = set(retrieved_ids[:k])
        return len(retrieved_at_k & relevant_ids) / k
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        retrieved_at_k = set(retrieved_ids[:k])
        return len(retrieved_at_k & relevant_ids) / len(relevant_ids)
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int) -> float:
        def dcg(scores: List[float]) -> float:
            return sum(score / np.log2(i + 2) for i, score in enumerate(scores))
        
        retrieved_scores = [relevance_scores.get(rid, 0.0) for rid in retrieved_ids[:k]]
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        
        dcg_val = dcg(retrieved_scores)
        idcg_val = dcg(ideal_scores)
        
        if idcg_val == 0:
            return 0.0
        
        return dcg_val / idcg_val


class ContextProblemMetrics:
    """Metrics for the 4 context engineering problems"""
    
    @staticmethod
    def poisoning_rate(memories: List[Any], current_time: datetime = None) -> float:
        if not memories:
            return 0.0
        
        if current_time is None:
            current_time = datetime.now()
        
        outdated_count = 0
        for mem in memories:
            age_days = (current_time - mem.timestamp).days
            
            if age_days > 90 and mem.confidence < 0.7:
                outdated_count += 1
            elif mem.confidence < 0.5:
                outdated_count += 1
        
        return outdated_count / len(memories)
    
    @staticmethod
    def distraction_score(memories: List[Any], query: str) -> float:
        if not memories:
            return 0.0
        
        query_terms = set(query.lower().split())
        total_tokens = 0
        irrelevant_tokens = 0
        
        for mem in memories:
            content_terms = set(mem.content.lower().split())
            tokens = len(mem.content.split())
            total_tokens += tokens
            
            overlap = len(query_terms & content_terms)
            if overlap == 0:
                irrelevant_tokens += tokens
            else:
                relevance_ratio = overlap / len(query_terms)
                if relevance_ratio < 0.2:
                    irrelevant_tokens += tokens * 0.5
        
        if total_tokens == 0:
            return 0.0
        
        return irrelevant_tokens / total_tokens
    
    @staticmethod
    def confusion_events(memories: List[Any]) -> int:
        confusion_count = 0
        
        entities = defaultdict(list)
        for mem in memories:
            entity = mem.metadata.get('entity', 'unknown')
            entities[entity].append(mem)
        
        for entity, mems in entities.items():
            if len(mems) > 1:
                contents = [m.content.lower() for m in mems]
                if len(set(contents)) > 1:
                    confusion_count += 1
        
        return confusion_count
    
    @staticmethod
    def clash_resolution_rate(conflicts_detected: int, conflicts_resolved: int) -> float:
        if conflicts_detected == 0:
            return 1.0
        return conflicts_resolved / conflicts_detected


class AgentPerformanceMetrics:
    """Metrics for agent performance with memory"""
    
    @staticmethod
    def task_success_rate(completed: int, total: int) -> float:
        if total == 0:
            return 0.0
        return completed / total
    
    @staticmethod
    def token_efficiency(useful_tokens: int, total_tokens: int) -> float:
        if total_tokens == 0:
            return 0.0
        return useful_tokens / total_tokens
    
    @staticmethod
    def response_quality_score(relevance: float, accuracy: float, completeness: float) -> float:
        return (relevance + accuracy + completeness) / 3.0


class MemoryManagementMetrics:
    """Metrics for memory lifecycle management"""
    
    @staticmethod
    def utilization_rate(accessed_count: int, total_count: int) -> float:
        if total_count == 0:
            return 0.0
        return accessed_count / total_count
    
    @staticmethod
    def pruning_accuracy(correctly_pruned: int, total_pruned: int) -> float:
        if total_pruned == 0:
            return 1.0
        return correctly_pruned / total_pruned
    
    @staticmethod
    def average_memory_age(memories: List[Any], current_time: datetime = None) -> float:
        if not memories:
            return 0.0
        
        if current_time is None:
            current_time = datetime.now()
        
        ages = [(current_time - mem.timestamp).days for mem in memories]
        return np.mean(ages)
    
    @staticmethod
    def memory_freshness_score(memories: List[Any], current_time: datetime = None) -> float:
        if not memories:
            return 0.0
        
        if current_time is None:
            current_time = datetime.now()
        
        freshness_scores = []
        for mem in memories:
            age_days = (current_time - mem.timestamp).days
            freshness = np.exp(-0.01 * age_days)
            freshness_scores.append(freshness)
        
        return np.mean(freshness_scores)


class MetricsAggregator:
    """Aggregate and track metrics over time"""
    
    def __init__(self):
        self.history = defaultdict(list)
    
    def record(self, metric_name: str, value: float):
        self.history[metric_name].append({
            'value': value,
            'timestamp': datetime.now()
        })
    
    def get_summary(self, metric_name: str, window_size: int = None) -> Dict[str, float]:
        if metric_name not in self.history:
            return {}
        
        values = [entry['value'] for entry in self.history[metric_name]]
        
        if window_size:
            values = values[-window_size:]
        
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
    
    def get_all_summaries(self, window_size: int = None) -> Dict[str, Dict[str, float]]:
        return {
            metric: self.get_summary(metric, window_size)
            for metric in self.history.keys()
        }
    
    def check_thresholds(self, thresholds: Dict[str, Tuple[str, float]]) -> List[str]:
        alerts = []
        
        for metric, (operator, threshold) in thresholds.items():
            summary = self.get_summary(metric)
            if not summary:
                continue
            
            current_value = summary['mean']
            
            if operator == '>' and current_value > threshold:
                alerts.append(f"⚠️ {metric} above threshold: {current_value:.3f} > {threshold}")
            elif operator == '<' and current_value < threshold:
                alerts.append(f"⚠️ {metric} below threshold: {current_value:.3f} < {threshold}")
            elif operator == '>=' and current_value >= threshold:
                alerts.append(f"⚠️ {metric} at/above threshold: {current_value:.3f} >= {threshold}")
            elif operator == '<=' and current_value <= threshold:
                alerts.append(f"⚠️ {metric} at/below threshold: {current_value:.3f} <= {threshold}")
        
        return alerts
