import sys
sys.path.insert(0, '/Users/jchen65/dev/ai_playground/agentic_memory')

from agentic_memory import AgenticMemory, MemoryType, MemoryScope
from evaluation.evaluator import MemoryEvaluator, GoldenDataset, ABTestEvaluator
import json


def demo_retrieval_evaluation():
    """Demonstrate retrieval quality evaluation"""
    print("=" * 70)
    print("DEMO: Retrieval Quality Evaluation")
    print("=" * 70)
    
    memory = AgenticMemory()
    
    print("\n1. Storing test memories...")
    mem1 = memory.store_semantic(
        entity="python",
        content="Python is excellent for data science, machine learning, and backend development",
        scope=MemoryScope.GLOBAL,
        importance=0.9,
        tags=["programming", "python"]
    )
    
    mem2 = memory.store_semantic(
        entity="javascript",
        content="JavaScript is essential for web development and runs in browsers",
        scope=MemoryScope.GLOBAL,
        importance=0.8,
        tags=["programming", "javascript"]
    )
    
    mem3 = memory.store_semantic(
        entity="rust",
        content="Rust is a systems programming language focused on safety and performance",
        scope=MemoryScope.GLOBAL,
        importance=0.7,
        tags=["programming", "rust"]
    )
    
    print(f"   ✓ Stored {len([mem1, mem2, mem3])} memories")
    
    print("\n2. Creating golden dataset...")
    dataset = GoldenDataset()
    dataset.add_query(
        query="What language is good for data science?",
        relevant_memory_ids=[mem1.id],
        relevance_scores={mem1.id: 1.0, mem2.id: 0.2, mem3.id: 0.3}
    )
    dataset.add_query(
        query="What should I use for web development?",
        relevant_memory_ids=[mem2.id],
        relevance_scores={mem1.id: 0.3, mem2.id: 1.0, mem3.id: 0.1}
    )
    print(f"   ✓ Created dataset with {len(dataset)} queries")
    
    print("\n3. Running evaluation...")
    evaluator = MemoryEvaluator(memory)
    results = evaluator.evaluate_retrieval_quality(dataset)
    
    print("\n📊 Retrieval Quality Metrics:")
    print(f"   Precision:    {results['precision']:.3f}")
    print(f"   Recall:       {results['recall']:.3f}")
    print(f"   F1 Score:     {results['f1']:.3f}")
    print(f"   MRR:          {results['mrr']:.3f}")
    print(f"   Precision@5:  {results['precision@5']:.3f}")
    print(f"   Recall@10:    {results['recall@10']:.3f}")
    print(f"   NDCG@10:      {results['ndcg@10']:.3f}")
    
    print("\n✅ Target Benchmarks:")
    print(f"   Precision > 0.70: {'✓' if results['precision'] > 0.70 else '✗'}")
    print(f"   Recall > 0.60:    {'✓' if results['recall'] > 0.60 else '✗'}")
    print(f"   MRR > 0.60:       {'✓' if results['mrr'] > 0.60 else '✗'}")


def demo_context_problems_evaluation():
    """Demonstrate context engineering problems evaluation"""
    print("\n" + "=" * 70)
    print("DEMO: Context Engineering Problems Evaluation")
    print("=" * 70)
    
    memory = AgenticMemory()
    
    print("\n1. Creating test scenario with context problems...")
    
    from datetime import datetime, timedelta
    
    old_mem = memory.store_semantic(
        entity="api_info",
        content="API endpoint is /api/v1/users - this is outdated",
        scope=MemoryScope.GLOBAL,
        confidence=0.5
    )
    old_mem.timestamp = datetime.now() - timedelta(days=120)
    
    memory.store_semantic(
        entity="api_info",
        content="API endpoint is /api/v2/users - current version",
        scope=MemoryScope.GLOBAL,
        confidence=1.0
    )
    
    memory.store_episodic(
        source="system",
        content="Random log entry about system health check passing",
        scope=MemoryScope.SESSION,
        importance=0.1
    )
    
    memory.store_semantic(
        entity="user_pref",
        content="User prefers detailed explanations",
        scope=MemoryScope.GLOBAL
    )
    
    print("   ✓ Created memories with potential issues")
    
    print("\n2. Running context problems evaluation...")
    evaluator = MemoryEvaluator(memory)
    
    test_queries = [
        "What is the API endpoint?",
        "What are user preferences?"
    ]
    
    results = evaluator.evaluate_context_problems(test_queries)
    
    print("\n📊 Context Problem Metrics:")
    print(f"   Poisoning Rate:        {results['poisoning_rate']:.3f} (target: < 0.05)")
    print(f"   Distraction Score:     {results['distraction_score']:.3f} (target: < 0.20)")
    print(f"   Confusion Events:      {results['confusion_events']} (target: 0)")
    print(f"   Clash Resolution Rate: {results['clash_resolution_rate']:.3f} (target: > 0.90)")
    
    print("\n✅ Problem Detection:")
    print(f"   Context Poisoning:  {'⚠️ Detected' if results['poisoning_rate'] > 0.05 else '✓ Clean'}")
    print(f"   Context Distraction: {'⚠️ High' if results['distraction_score'] > 0.20 else '✓ Low'}")
    print(f"   Context Confusion:   {'⚠️ Found' if results['confusion_events'] > 0 else '✓ None'}")


def demo_performance_evaluation():
    """Demonstrate performance metrics evaluation"""
    print("\n" + "=" * 70)
    print("DEMO: Performance Evaluation")
    print("=" * 70)
    
    memory = AgenticMemory()
    
    print("\n1. Populating memory store...")
    for i in range(20):
        memory.store_semantic(
            entity=f"fact_{i}",
            content=f"This is fact number {i} about various technical topics and concepts",
            scope=MemoryScope.GLOBAL,
            importance=0.5 + (i % 5) * 0.1
        )
    print("   ✓ Stored 20 memories")
    
    print("\n2. Running performance tests...")
    evaluator = MemoryEvaluator(memory)
    
    test_cases = [
        {'query': 'Tell me about fact 5'},
        {'query': 'What technical topics do you know?'},
        {'query': 'Give me information about concepts'},
        {'query': 'What facts are available?'},
        {'query': 'Tell me about various topics'}
    ]
    
    results = evaluator.evaluate_performance(test_cases)
    
    print("\n📊 Performance Metrics:")
    print(f"   Latency (mean):  {results['latency_mean']:.2f} ms")
    print(f"   Latency (p50):   {results['latency_p50']:.2f} ms")
    print(f"   Latency (p95):   {results['latency_p95']:.2f} ms")
    print(f"   Latency (p99):   {results['latency_p99']:.2f} ms")
    print(f"   Tokens (mean):   {results['tokens_mean']:.0f}")
    print(f"   Tokens (p95):    {results['tokens_p95']:.0f}")
    
    print("\n✅ Performance Targets:")
    print(f"   Latency p95 < 300ms: {'✓' if results['latency_p95'] < 300 else '✗'}")
    print(f"   Tokens efficient:    {'✓' if results['tokens_mean'] < 2000 else '✗'}")


def demo_memory_management_evaluation():
    """Demonstrate memory management evaluation"""
    print("\n" + "=" * 70)
    print("DEMO: Memory Management Evaluation")
    print("=" * 70)
    
    memory = AgenticMemory(enable_auto_pruning=False)
    
    print("\n1. Creating diverse memory set...")
    for i in range(10):
        memory.store_semantic(
            entity=f"entity_{i}",
            content=f"Content for entity {i}",
            scope=MemoryScope.GLOBAL,
            importance=0.3 + (i % 7) * 0.1
        )
    print("   ✓ Stored 10 memories")
    
    print("\n2. Simulating usage patterns...")
    memory.retrieve("Content for entity 0")
    memory.retrieve("Content for entity 1")
    memory.retrieve("Content for entity 2")
    print("   ✓ Accessed 3 memories")
    
    print("\n3. Evaluating memory management...")
    evaluator = MemoryEvaluator(memory)
    results = evaluator.evaluate_memory_management()
    
    print("\n📊 Memory Management Metrics:")
    print(f"   Total Memories:     {results['total_memories']}")
    print(f"   Accessed Memories:  {results['accessed_memories']}")
    print(f"   Utilization Rate:   {results['utilization_rate']:.3f} (target: > 0.25)")
    print(f"   Avg Age (days):     {results['avg_age_days']:.2f}")
    print(f"   Freshness Score:    {results['freshness_score']:.3f}")
    
    print("\n✅ Management Health:")
    print(f"   Utilization: {'✓ Good' if results['utilization_rate'] > 0.25 else '⚠️ Low'}")
    print(f"   Freshness:   {'✓ Fresh' if results['freshness_score'] > 0.8 else '⚠️ Aging'}")


def demo_ab_testing():
    """Demonstrate A/B testing framework"""
    print("\n" + "=" * 70)
    print("DEMO: A/B Testing (Memory vs No Memory)")
    print("=" * 70)
    
    memory = AgenticMemory()
    
    print("\n1. Setting up test scenario...")
    memory.store_semantic(
        entity="capital_france",
        content="The capital of France is Paris",
        scope=MemoryScope.GLOBAL
    )
    memory.store_semantic(
        entity="capital_japan",
        content="The capital of Japan is Tokyo",
        scope=MemoryScope.GLOBAL
    )
    memory.store_semantic(
        entity="capital_uk",
        content="The capital of United Kingdom is London",
        scope=MemoryScope.GLOBAL
    )
    print("   ✓ Stored knowledge base")
    
    print("\n2. Defining test agents...")
    
    def agent_with_memory(query, mem):
        result = mem.retrieve(query, task_type="factual")
        if result.memories:
            return f"Answer: {result.memories[0].content}"
        return "I don't know"
    
    def agent_without_memory(query):
        return "I don't have that information"
    
    print("   ✓ Agents defined")
    
    print("\n3. Running A/B test...")
    test_cases = [
        {
            'query': 'What is the capital of France?',
            'is_correct': lambda x: 'Paris' in x,
            'rate_quality': lambda x: 5.0 if 'Paris' in x else 1.0
        },
        {
            'query': 'What is the capital of Japan?',
            'is_correct': lambda x: 'Tokyo' in x,
            'rate_quality': lambda x: 5.0 if 'Tokyo' in x else 1.0
        },
        {
            'query': 'What is the capital of UK?',
            'is_correct': lambda x: 'London' in x,
            'rate_quality': lambda x: 5.0 if 'London' in x else 1.0
        }
    ]
    
    ab_evaluator = ABTestEvaluator(memory)
    results = ab_evaluator.run_ab_test(test_cases, agent_with_memory, agent_without_memory)
    
    print("\n📊 A/B Test Results:")
    print(f"   Success Rate (with memory):    {results['success_rate_with_memory']:.3f}")
    print(f"   Success Rate (without memory): {results['success_rate_without_memory']:.3f}")
    print(f"   Success Rate Improvement:      {results['success_rate_improvement']:.3f}")
    print(f"   Quality (with memory):         {results['quality_with_memory']:.2f}/5.0")
    print(f"   Quality (without memory):      {results['quality_without_memory']:.2f}/5.0")
    print(f"   Quality Improvement:           {results['quality_improvement']:.2f}")
    print(f"   Latency Overhead:              {results['latency_overhead']*1000:.2f} ms")
    
    print("\n✅ Memory Impact:")
    improvement_pct = results['success_rate_improvement'] * 100
    print(f"   Task Success: +{improvement_pct:.1f}%")
    print(f"   Response Quality: +{results['quality_improvement']:.1f} points")


def demo_full_evaluation():
    """Demonstrate complete evaluation pipeline"""
    print("\n" + "=" * 70)
    print("DEMO: Full Evaluation Pipeline")
    print("=" * 70)
    
    memory = AgenticMemory()
    
    print("\n1. Setting up comprehensive test scenario...")
    mem1 = memory.store_semantic(
        entity="python",
        content="Python is excellent for data science, AI, and backend development",
        scope=MemoryScope.GLOBAL,
        importance=0.9
    )
    
    mem2 = memory.store_procedural(
        workflow_name="debugging",
        content="Debugging workflow: 1) Reproduce issue, 2) Check logs, 3) Add tests, 4) Fix and verify",
        scope=MemoryScope.PROJECT,
        importance=0.8
    )
    
    mem3 = memory.store_episodic(
        source="user",
        content="User is working on a machine learning project using scikit-learn",
        scope=MemoryScope.SESSION,
        importance=0.7
    )
    
    print("   ✓ Created test memories")
    
    print("\n2. Creating evaluation dataset...")
    dataset = GoldenDataset()
    dataset.add_query("What language is good for AI?", [mem1.id])
    dataset.add_query("How do I debug issues?", [mem2.id])
    
    test_queries = [
        "What language should I use?",
        "What is the debugging process?",
        "What project is the user working on?"
    ]
    
    print("   ✓ Dataset ready")
    
    print("\n3. Running full evaluation...")
    evaluator = MemoryEvaluator(memory)
    results = evaluator.run_full_evaluation(dataset, test_queries)
    
    print("\n📊 Complete Evaluation Results:")
    print("\n   Retrieval Quality:")
    for metric, value in results['retrieval'].items():
        print(f"      {metric:15s}: {value:.3f}")
    
    print("\n   Context Problems:")
    for metric, value in results['context_problems'].items():
        print(f"      {metric:25s}: {value:.3f}")
    
    print("\n   Performance:")
    for metric, value in results['performance'].items():
        print(f"      {metric:20s}: {value:.2f}")
    
    print("\n   Memory Management:")
    for metric, value in results['memory_management'].items():
        if isinstance(value, float):
            print(f"      {metric:20s}: {value:.3f}")
        else:
            print(f"      {metric:20s}: {value}")
    
    print("\n4. Checking system health...")
    is_healthy, alerts = evaluator.check_health()
    
    print(f"\n   System Health: {'✅ HEALTHY' if is_healthy else '⚠️ ISSUES DETECTED'}")
    if alerts:
        print("\n   Alerts:")
        for alert in alerts:
            print(f"      {alert}")
    else:
        print("   No alerts - all metrics within target ranges")


def main():
    """Run all evaluation demonstrations"""
    demos = [
        demo_retrieval_evaluation,
        demo_context_problems_evaluation,
        demo_performance_evaluation,
        demo_memory_management_evaluation,
        demo_ab_testing,
        demo_full_evaluation
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n❌ Error in {demo.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("All evaluation demonstrations complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  • Run 'pytest tests/test_evaluation.py -v' for automated tests")
    print("  • Check EVALUATION.md for detailed methodology")
    print("  • Integrate evaluation into your CI/CD pipeline")


if __name__ == "__main__":
    main()
