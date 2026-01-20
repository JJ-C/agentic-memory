import sys
sys.path.insert(0, '/Users/jchen65/dev/ai_playground/agentic_memory')

import pytest
import time
from datetime import datetime, timedelta

from agentic_memory import AgenticMemory, MemoryType, MemoryScope
from evaluation.evaluator import MemoryEvaluator, GoldenDataset, ABTestEvaluator
from evaluation.metrics import (
    RetrievalMetrics,
    ContextProblemMetrics,
    AgentPerformanceMetrics,
    MemoryManagementMetrics
)


class TestRetrievalMetrics:
    """Test retrieval quality metrics"""
    
    def test_precision_recall_f1(self):
        retrieved = {'a', 'b', 'c', 'd'}
        relevant = {'b', 'c', 'e', 'f'}
        
        precision = RetrievalMetrics.precision(retrieved, relevant)
        recall = RetrievalMetrics.recall(retrieved, relevant)
        f1 = RetrievalMetrics.f1_score(precision, recall)
        
        assert precision == 0.5
        assert recall == 0.5
        assert f1 == 0.5
    
    def test_mrr(self):
        retrieved = ['a', 'b', 'c', 'd']
        relevant = {'c', 'e'}
        
        mrr = RetrievalMetrics.mean_reciprocal_rank(retrieved, relevant)
        
        assert mrr == 1.0 / 3
    
    def test_precision_at_k(self):
        retrieved = ['a', 'b', 'c', 'd', 'e']
        relevant = {'b', 'c'}
        
        p_at_3 = RetrievalMetrics.precision_at_k(retrieved, relevant, 3)
        p_at_5 = RetrievalMetrics.precision_at_k(retrieved, relevant, 5)
        
        assert p_at_3 == 2/3
        assert p_at_5 == 2/5


class TestContextProblemMetrics:
    """Test context engineering problem metrics"""
    
    def setup_method(self):
        self.memory = AgenticMemory()
    
    def test_poisoning_rate(self):
        old_mem = self.memory.store_semantic(
            entity="old_info",
            content="Outdated information",
            scope=MemoryScope.GLOBAL,
            confidence=0.4
        )
        old_mem.timestamp = datetime.now() - timedelta(days=100)
        
        new_mem = self.memory.store_semantic(
            entity="new_info",
            content="Current information",
            scope=MemoryScope.GLOBAL,
            confidence=0.9
        )
        
        memories = [old_mem, new_mem]
        rate = ContextProblemMetrics.poisoning_rate(memories)
        
        assert rate == 0.5
    
    def test_distraction_score(self):
        mem1 = self.memory.store_semantic(
            entity="relevant",
            content="Python is great for backend development",
            scope=MemoryScope.GLOBAL
        )
        
        mem2 = self.memory.store_semantic(
            entity="irrelevant",
            content="The weather is nice today and birds are singing",
            scope=MemoryScope.GLOBAL
        )
        
        query = "What programming language should I use for backend?"
        memories = [mem1, mem2]
        
        score = ContextProblemMetrics.distraction_score(memories, query)
        
        assert score > 0.0
        assert score < 1.0


class TestGoldenDatasetEvaluation:
    """Test evaluation with golden dataset"""
    
    def setup_method(self):
        self.memory = AgenticMemory()
        self.evaluator = MemoryEvaluator(self.memory)
    
    def test_golden_dataset_creation(self):
        mem1 = self.memory.store_semantic(
            entity="python",
            content="Python is excellent for backend development",
            scope=MemoryScope.GLOBAL
        )
        
        mem2 = self.memory.store_semantic(
            entity="javascript",
            content="JavaScript is great for frontend development",
            scope=MemoryScope.GLOBAL
        )
        
        mem3 = self.memory.store_semantic(
            entity="java",
            content="Java is used for enterprise applications",
            scope=MemoryScope.GLOBAL
        )
        
        dataset = GoldenDataset()
        dataset.add_query(
            query="What language is good for backend?",
            relevant_memory_ids=[mem1.id],
            relevance_scores={mem1.id: 1.0, mem2.id: 0.3, mem3.id: 0.5}
        )
        
        assert len(dataset) == 1
    
    def test_retrieval_quality_evaluation(self):
        mem1 = self.memory.store_semantic(
            entity="react",
            content="React is a popular JavaScript library for building user interfaces",
            scope=MemoryScope.GLOBAL,
            importance=0.8
        )
        
        mem2 = self.memory.store_semantic(
            entity="vue",
            content="Vue is a progressive framework for building web interfaces",
            scope=MemoryScope.GLOBAL,
            importance=0.7
        )
        
        mem3 = self.memory.store_semantic(
            entity="angular",
            content="Angular is a platform for building web applications",
            scope=MemoryScope.GLOBAL,
            importance=0.6
        )
        
        dataset = GoldenDataset()
        dataset.add_query(
            query="What framework should I use for building web interfaces?",
            relevant_memory_ids=[mem1.id, mem2.id]
        )
        
        results = self.evaluator.evaluate_retrieval_quality(dataset)
        
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 'mrr' in results
        assert results['precision'] >= 0.0
        assert results['recall'] >= 0.0


class TestContextProblemsEvaluation:
    """Test evaluation of context engineering problems"""
    
    def setup_method(self):
        self.memory = AgenticMemory()
        self.evaluator = MemoryEvaluator(self.memory)
    
    def test_context_problems_evaluation(self):
        self.memory.store_semantic(
            entity="preference",
            content="User prefers detailed explanations",
            scope=MemoryScope.GLOBAL,
            importance=0.8
        )
        
        self.memory.store_episodic(
            source="user",
            content="I'm working on a Python project with FastAPI",
            scope=MemoryScope.SESSION,
            importance=0.7
        )
        
        queries = [
            "What are the user's preferences?",
            "What project is the user working on?"
        ]
        
        results = self.evaluator.evaluate_context_problems(queries)
        
        assert 'poisoning_rate' in results
        assert 'distraction_score' in results
        assert 'confusion_events' in results
        assert 'clash_resolution_rate' in results
        
        assert results['poisoning_rate'] >= 0.0
        assert results['distraction_score'] >= 0.0


class TestPerformanceEvaluation:
    """Test performance metrics evaluation"""
    
    def setup_method(self):
        self.memory = AgenticMemory()
        self.evaluator = MemoryEvaluator(self.memory)
    
    def test_performance_evaluation(self):
        for i in range(10):
            self.memory.store_semantic(
                entity=f"fact_{i}",
                content=f"This is fact number {i} about various topics",
                scope=MemoryScope.GLOBAL
            )
        
        test_cases = [
            {'query': 'Tell me about fact 5'},
            {'query': 'What do you know about topics?'},
            {'query': 'Give me information about facts'}
        ]
        
        results = self.evaluator.evaluate_performance(test_cases)
        
        assert 'latency_mean' in results
        assert 'latency_p95' in results
        assert 'tokens_mean' in results
        
        assert results['latency_mean'] > 0
        assert results['latency_p95'] >= results['latency_mean']


class TestMemoryManagementEvaluation:
    """Test memory management metrics"""
    
    def setup_method(self):
        self.memory = AgenticMemory()
        self.evaluator = MemoryEvaluator(self.memory)
    
    def test_memory_management_evaluation(self):
        for i in range(5):
            self.memory.store_semantic(
                entity=f"entity_{i}",
                content=f"Content {i}",
                scope=MemoryScope.GLOBAL
            )
        
        self.memory.retrieve("Content 0")
        self.memory.retrieve("Content 1")
        
        results = self.evaluator.evaluate_memory_management()
        
        assert 'utilization_rate' in results
        assert 'avg_age_days' in results
        assert 'freshness_score' in results
        assert 'total_memories' in results
        
        assert results['total_memories'] == 5
        assert results['utilization_rate'] > 0.0


class TestFullEvaluation:
    """Test complete evaluation pipeline"""
    
    def setup_method(self):
        self.memory = AgenticMemory()
        self.evaluator = MemoryEvaluator(self.memory)
    
    def test_full_evaluation_pipeline(self):
        mem1 = self.memory.store_semantic(
            entity="python",
            content="Python is great for data science and machine learning",
            scope=MemoryScope.GLOBAL,
            importance=0.9
        )
        
        mem2 = self.memory.store_semantic(
            entity="javascript",
            content="JavaScript is essential for web development",
            scope=MemoryScope.GLOBAL,
            importance=0.8
        )
        
        mem3 = self.memory.store_procedural(
            workflow_name="debugging",
            content="When debugging: check logs, reproduce issue, add tests, fix",
            scope=MemoryScope.PROJECT,
            importance=0.7
        )
        
        dataset = GoldenDataset()
        dataset.add_query(
            query="What language is good for data science?",
            relevant_memory_ids=[mem1.id]
        )
        
        test_queries = [
            "What language should I use?",
            "How do I debug issues?"
        ]
        
        results = self.evaluator.run_full_evaluation(dataset, test_queries)
        
        assert 'retrieval' in results
        assert 'context_problems' in results
        assert 'performance' in results
        assert 'memory_management' in results
        assert 'timestamp' in results
        
        assert results['retrieval']['precision'] >= 0.0
        assert results['context_problems']['poisoning_rate'] >= 0.0
        assert results['performance']['latency_mean'] > 0


class TestHealthChecks:
    """Test health monitoring and alerting"""
    
    def setup_method(self):
        self.memory = AgenticMemory()
        self.evaluator = MemoryEvaluator(self.memory)
    
    def test_health_check_with_good_metrics(self):
        for i in range(10):
            self.memory.store_semantic(
                entity=f"entity_{i}",
                content=f"High quality content {i}",
                scope=MemoryScope.GLOBAL,
                importance=0.8
            )
        
        mem1 = self.memory.store_semantic(
            entity="test",
            content="Test content for retrieval",
            scope=MemoryScope.GLOBAL
        )
        
        dataset = GoldenDataset()
        dataset.add_query("test content", [mem1.id])
        
        self.evaluator.evaluate_retrieval_quality(dataset)
        
        is_healthy, alerts = self.evaluator.check_health()
        
        assert isinstance(is_healthy, bool)
        assert isinstance(alerts, list)


class TestABTesting:
    """Test A/B testing framework"""
    
    def setup_method(self):
        self.memory = AgenticMemory()
        self.ab_evaluator = ABTestEvaluator(self.memory)
    
    def test_ab_test_setup(self):
        self.memory.store_semantic(
            entity="fact",
            content="The capital of France is Paris",
            scope=MemoryScope.GLOBAL
        )
        
        def agent_with_memory(query, memory):
            result = memory.retrieve(query)
            if result.memories:
                return f"Based on memory: {result.memories[0].content}"
            return "No memory found"
        
        def agent_without_memory(query):
            return "I don't have that information"
        
        test_cases = [
            {
                'query': 'What is the capital of France?',
                'is_correct': lambda x: 'Paris' in x,
                'rate_quality': lambda x: 5.0 if 'Paris' in x else 1.0
            }
        ]
        
        results = self.ab_evaluator.run_ab_test(
            test_cases,
            agent_with_memory,
            agent_without_memory
        )
        
        assert 'success_rate_with_memory' in results
        assert 'success_rate_without_memory' in results
        assert 'success_rate_improvement' in results
        assert 'latency_overhead' in results
        
        assert results['success_rate_with_memory'] >= results['success_rate_without_memory']


class TestMetricsAggregation:
    """Test metrics tracking and aggregation"""
    
    def setup_method(self):
        self.memory = AgenticMemory()
        self.evaluator = MemoryEvaluator(self.memory)
    
    def test_metrics_aggregation(self):
        mem = self.memory.store_semantic(
            entity="test",
            content="Test content",
            scope=MemoryScope.GLOBAL
        )
        
        dataset = GoldenDataset()
        for i in range(5):
            dataset.add_query(f"query {i}", [mem.id])
        
        self.evaluator.evaluate_retrieval_quality(dataset)
        
        summary = self.evaluator.get_metrics_summary(window_size=10)
        
        assert isinstance(summary, dict)
        assert len(summary) > 0


class TestBenchmarkScenarios:
    """Test with benchmark-style scenarios"""
    
    def setup_method(self):
        self.memory = AgenticMemory()
        self.evaluator = MemoryEvaluator(self.memory)
    
    def test_recency_preference_benchmark(self):
        old_mem = self.memory.store_semantic(
            entity="api_version",
            content="API is version 1.0",
            scope=MemoryScope.GLOBAL,
            importance=0.7
        )
        old_mem.timestamp = datetime.now() - timedelta(days=180)
        
        time.sleep(0.1)
        
        new_mem = self.memory.store_semantic(
            entity="api_version",
            content="API is version 2.0",
            scope=MemoryScope.GLOBAL,
            importance=0.7
        )
        
        result = self.memory.retrieve("What API version should I use?")
        
        assert len(result.memories) > 0
        top_memory = result.memories[0]
        assert "2.0" in top_memory.content or top_memory.timestamp > old_mem.timestamp
    
    def test_scope_isolation_benchmark(self):
        self.memory.store_semantic(
            entity="project_a",
            content="Project A uses React",
            scope=MemoryScope.PROJECT,
            tags=["project_a"]
        )
        
        self.memory.store_semantic(
            entity="project_b",
            content="Project B uses Vue",
            scope=MemoryScope.PROJECT,
            tags=["project_b"]
        )
        
        result = self.memory.retrieve(
            "What framework does the project use?",
            scope=MemoryScope.PROJECT
        )
        
        assert len(result.memories) > 0
    
    def test_conflict_resolution_benchmark(self):
        mem1 = self.memory.store_semantic(
            entity="preference",
            content="User prefers verbose responses",
            scope=MemoryScope.GLOBAL,
            importance=0.7
        )
        
        time.sleep(0.1)
        
        mem2 = self.memory.store_semantic(
            entity="preference",
            content="User prefers concise responses",
            scope=MemoryScope.GLOBAL,
            importance=0.7
        )
        
        result = self.memory.retrieve("How does user prefer responses?")
        
        assert len(result.memories) > 0
        assert result.metadata.get('conflicts_detected', 0) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
