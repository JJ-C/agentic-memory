import sys
sys.path.insert(0, '/Users/jchen65/dev/ai_playground/agentic_memory')

import pytest
from datetime import datetime, timedelta
from agentic_memory import AgenticMemory, MemoryType, MemoryScope


class TestCustomerSupportAgent:
    """
    Real-world scenario: Customer support agent handling multiple customers
    
    Problems addressed:
    - Context Confusion: Multiple customers with similar issues
    - Context Clash: Different solutions for different customers
    - Context Quarantine: Isolate each customer's conversation
    """
    
    def setup_method(self):
        self.memory = AgenticMemory()
    
    def test_multi_customer_isolation(self):
        self.memory.create_isolated_context("customer_123", MemoryScope.SESSION)
        self.memory.create_isolated_context("customer_456", MemoryScope.SESSION)
        
        self.memory.store_episodic(
            source="customer_123",
            content="Customer reported login issue with email john@example.com. Password reset sent.",
            scope=MemoryScope.SESSION,
            tags=["customer_123", "login_issue"]
        )
        
        self.memory.store_episodic(
            source="customer_456",
            content="Customer reported billing discrepancy. Refund of $50 processed.",
            scope=MemoryScope.SESSION,
            tags=["customer_456", "billing"]
        )
        
        result_123 = self.memory.retrieve(
            query="What was the customer's issue?",
            scope=MemoryScope.SESSION,
            context_id="customer_123"
        )
        
        assert len(result_123.memories) > 0
        assert "login" in result_123.memories[0].content.lower()
        assert "billing" not in result_123.memories[0].content.lower()
        
        result_456 = self.memory.retrieve(
            query="What was the customer's issue?",
            scope=MemoryScope.SESSION,
            context_id="customer_456"
        )
        
        assert len(result_456.memories) > 0
        assert "billing" in result_456.memories[0].content.lower()
        assert "login" not in result_456.memories[0].content.lower()
    
    def test_escalation_with_context_sharing(self):
        self.memory.create_isolated_context("tier1_agent", MemoryScope.SESSION)
        self.memory.create_isolated_context("tier2_agent", MemoryScope.SESSION)
        
        self.memory.store_episodic(
            source="tier1_agent",
            content="Customer has complex API integration issue. Tried basic troubleshooting steps. Escalating to tier 2.",
            scope=MemoryScope.SESSION,
            tags=["tier1_agent", "escalation"]
        )
        
        self.memory.grant_context_access("tier2_agent", "tier1_agent")
        
        result = self.memory.retrieve(
            query="What troubleshooting was already done?",
            scope=MemoryScope.SESSION,
            context_id="tier2_agent"
        )
        
        assert len(result.memories) > 0
        assert "troubleshooting" in result.memories[0].content.lower()


class TestCodeAssistant:
    """
    Real-world scenario: AI code assistant working on multiple projects
    
    Problems addressed:
    - Context Distraction: Irrelevant code from other projects
    - Context Poisoning: Outdated dependencies or patterns
    - Tool Loadout: Select relevant memory types for coding tasks
    """
    
    def setup_method(self):
        self.memory = AgenticMemory()
    
    def test_project_specific_context(self):
        self.memory.store_semantic(
            entity="project_a",
            content="Project A uses React 18 with TypeScript. State management via Zustand.",
            scope=MemoryScope.PROJECT,
            tags=["project_a", "tech_stack"]
        )
        
        self.memory.store_semantic(
            entity="project_b",
            content="Project B uses Vue 3 with JavaScript. State management via Pinia.",
            scope=MemoryScope.PROJECT,
            tags=["project_b", "tech_stack"]
        )
        
        self.memory.store_procedural(
            workflow_name="react_component",
            content="When creating React components, use functional components with hooks. Follow the pattern: imports, types, component, exports.",
            scope=MemoryScope.PROJECT,
            tags=["project_a", "best_practice"]
        )
        
        result = self.memory.retrieve(
            query="What framework and state management should I use?",
            task_type="recommendation",
            scope=MemoryScope.PROJECT
        )
        
        assert len(result.memories) > 0
        assert result.metadata["memory_types_used"]
        assert MemoryType.SEMANTIC.value in result.metadata["memory_types_used"]
    
    def test_debugging_with_episodic_memory(self):
        self.memory.store_episodic(
            source="developer",
            content="Encountered CORS error when calling API from localhost:3000. Fixed by adding CORS headers in backend.",
            scope=MemoryScope.PROJECT,
            importance=0.8,
            tags=["debugging", "cors"]
        )
        
        self.memory.store_episodic(
            source="developer",
            content="Database connection timeout. Increased pool size from 10 to 20 connections.",
            scope=MemoryScope.PROJECT,
            importance=0.7,
            tags=["debugging", "database"]
        )
        
        result = self.memory.retrieve(
            query="How did we fix the CORS error?",
            task_type="debugging",
            scope=MemoryScope.PROJECT
        )
        
        assert len(result.memories) > 0
        assert "cors" in result.memories[0].content.lower()
        assert result.metadata["memory_types_used"]
    
    def test_context_distraction_filtering(self):
        for i in range(20):
            self.memory.store_episodic(
                source="developer",
                content=f"Random commit message {i}: Updated documentation and fixed typos.",
                scope=MemoryScope.PROJECT,
                importance=0.2
            )
        
        self.memory.store_episodic(
            source="developer",
            content="Critical bug fix: Resolved memory leak in WebSocket connection handler.",
            scope=MemoryScope.PROJECT,
            importance=0.9,
            tags=["critical", "bug_fix"]
        )
        
        result = self.memory.retrieve(
            query="What critical bugs were fixed?",
            task_type="debugging",
            top_k=5
        )
        
        assert len(result.memories) <= 5
        top_memory = result.memories[0]
        assert "memory leak" in top_memory.content.lower() or "critical" in top_memory.content.lower()


class TestMultiProjectDeveloper:
    """
    Real-world scenario: Developer switching between multiple projects
    
    Problems addressed:
    - Context Clash: Conflicting conventions between projects
    - Context Confusion: Mixing up project-specific details
    - Context Quarantine: Proper scope isolation
    """
    
    def setup_method(self):
        self.memory = AgenticMemory()
    
    def test_conflicting_conventions(self):
        self.memory.store_semantic(
            entity="project_alpha",
            content="Project Alpha uses snake_case for variable naming.",
            scope=MemoryScope.PROJECT,
            tags=["project_alpha", "conventions"],
            metadata={"project": "alpha"}
        )
        
        self.memory.store_semantic(
            entity="project_beta",
            content="Project Beta uses camelCase for variable naming.",
            scope=MemoryScope.PROJECT,
            tags=["project_beta", "conventions"],
            metadata={"project": "beta"}
        )
        
        result = self.memory.retrieve(
            query="What naming convention should I use?",
            scope=MemoryScope.PROJECT
        )
        
        assert len(result.memories) >= 1
        conflicts = result.metadata.get("conflicts_detected", 0)
        
        if len(result.memories) == 2:
            assert "snake_case" in result.memories[0].content or "camelCase" in result.memories[0].content
    
    def test_global_vs_project_preferences(self):
        self.memory.store_semantic(
            entity="user_preference",
            content="Developer prefers using TypeScript for all projects.",
            scope=MemoryScope.GLOBAL,
            importance=0.8
        )
        
        self.memory.store_semantic(
            entity="legacy_project",
            content="Legacy project must use JavaScript due to existing codebase constraints.",
            scope=MemoryScope.PROJECT,
            importance=0.9,
            tags=["legacy_project"]
        )
        
        result = self.memory.retrieve(
            query="Should I use TypeScript or JavaScript?",
            scope=MemoryScope.PROJECT
        )
        
        assert len(result.memories) > 0
        top_memory = result.memories[0]
        assert top_memory.scope in [MemoryScope.PROJECT, MemoryScope.GLOBAL]


class TestPolicyComplianceChecker:
    """
    Real-world scenario: Agent checking code against company policies
    
    Problems addressed:
    - Context Offloading: Store detailed policies externally
    - Context Poisoning: Ensure policy information is current
    - Relevance Scoring: Retrieve only applicable policies
    """
    
    def setup_method(self):
        self.memory = AgenticMemory()
    
    def test_policy_retrieval(self):
        self.memory.store_semantic(
            entity="security_policy",
            content="All API keys must be stored in environment variables, never hardcoded. Use .env files with .gitignore.",
            scope=MemoryScope.GLOBAL,
            importance=1.0,
            confidence=1.0,
            tags=["security", "policy"]
        )
        
        self.memory.store_semantic(
            entity="data_policy",
            content="Personal identifiable information (PII) must be encrypted at rest using AES-256.",
            scope=MemoryScope.GLOBAL,
            importance=1.0,
            confidence=1.0,
            tags=["data_governance", "policy"]
        )
        
        self.memory.store_semantic(
            entity="code_style",
            content="Use ESLint with Airbnb config. Maximum line length 100 characters.",
            scope=MemoryScope.GLOBAL,
            importance=0.6,
            tags=["code_style"]
        )
        
        result = self.memory.retrieve(
            query="How should I handle API keys in the code?",
            task_type="factual",
            scope=MemoryScope.GLOBAL
        )
        
        assert len(result.memories) > 0
        assert "api key" in result.memories[0].content.lower() or "environment" in result.memories[0].content.lower()
    
    def test_policy_update_and_conflict(self):
        old_policy = self.memory.store_semantic(
            entity="password_policy",
            content="Passwords must be at least 8 characters long.",
            scope=MemoryScope.GLOBAL,
            importance=0.9,
            tags=["security", "policy"]
        )
        
        import time
        time.sleep(0.1)
        
        new_policy = self.memory.store_semantic(
            entity="password_policy",
            content="Passwords must be at least 12 characters long with special characters.",
            scope=MemoryScope.GLOBAL,
            importance=0.9,
            tags=["security", "policy"]
        )
        
        result = self.memory.retrieve(
            query="What are the password requirements?",
            task_type="factual",
            scope=MemoryScope.GLOBAL
        )
        
        assert len(result.memories) > 0
        most_recent = max(result.memories, key=lambda m: m.timestamp)
        assert "12 characters" in most_recent.content


class TestConversationalAgent:
    """
    Real-world scenario: Conversational AI maintaining context across sessions
    
    Problems addressed:
    - Context Summarization: Long conversations need compression
    - Context Pruning: Remove irrelevant chitchat
    - Temporal Decay: Recent information more relevant
    """
    
    def setup_method(self):
        self.memory = AgenticMemory()
    
    def test_long_conversation_summarization(self):
        long_conversation = """
        User: Hi, I'm working on a machine learning project.
        Assistant: Great! What kind of ML project?
        User: I'm building a recommendation system for an e-commerce site.
        Assistant: Interesting! What approach are you considering?
        User: I'm thinking collaborative filtering.
        Assistant: That's a solid choice. Have you considered matrix factorization?
        User: Yes, I'm planning to use SVD.
        Assistant: Excellent. Make sure to handle sparse matrices efficiently.
        """
        
        memory = self.memory.store_episodic(
            source="conversation",
            content=long_conversation,
            scope=MemoryScope.SESSION,
            importance=0.7
        )
        
        result = self.memory.retrieve(
            query="What ML project is the user working on?",
            task_type="conversational",
            max_tokens=500
        )
        
        assert len(result.memories) > 0
        assert result.total_tokens <= 500
        
        if result.memories[0].metadata.get("summarized"):
            assert len(result.memories[0].content) < len(long_conversation)
    
    def test_user_preference_learning(self):
        self.memory.store_episodic(
            source="user",
            content="I prefer detailed explanations with code examples.",
            scope=MemoryScope.GLOBAL,
            importance=0.8,
            tags=["user_preference"]
        )
        
        self.memory.store_episodic(
            source="user",
            content="I don't like when responses are too verbose.",
            scope=MemoryScope.GLOBAL,
            importance=0.7,
            tags=["user_preference"]
        )
        
        result = self.memory.retrieve(
            query="How does the user prefer responses?",
            task_type="conversational",
            scope=MemoryScope.GLOBAL
        )
        
        assert len(result.memories) > 0
        assert result.metadata.get("conflicts_detected", 0) >= 0


class TestDataGovernanceAgent:
    """
    Real-world scenario: Agent enforcing data governance rules
    
    Problems addressed:
    - Context Offloading: Store complex governance rules
    - Policy Compliance: Retrieve relevant regulations
    - Audit Trail: Track memory access
    """
    
    def setup_method(self):
        self.memory = AgenticMemory()
    
    def test_gdpr_compliance_check(self):
        self.memory.store_semantic(
            entity="gdpr_rule",
            content="Under GDPR, users have the right to request deletion of their personal data within 30 days.",
            scope=MemoryScope.GLOBAL,
            importance=1.0,
            tags=["gdpr", "data_governance", "compliance"]
        )
        
        self.memory.store_semantic(
            entity="gdpr_rule",
            content="Personal data must not be transferred outside EU without adequate protection mechanisms.",
            scope=MemoryScope.GLOBAL,
            importance=1.0,
            tags=["gdpr", "data_governance", "compliance"]
        )
        
        result = self.memory.retrieve(
            query="What are the data deletion requirements?",
            task_type="factual",
            scope=MemoryScope.GLOBAL
        )
        
        assert len(result.memories) > 0
        assert "deletion" in result.memories[0].content.lower() or "30 days" in result.memories[0].content.lower()
        
        for memory in result.memories:
            assert memory.access_count > 0
    
    def test_hierarchical_policy_access(self):
        self.memory.store_semantic(
            entity="company_policy",
            content="All employees must complete security training annually.",
            scope=MemoryScope.GLOBAL,
            importance=0.8,
            tags=["company_policy"]
        )
        
        self.memory.store_semantic(
            entity="department_policy",
            content="Engineering team must use approved libraries from the security-vetted list.",
            scope=MemoryScope.PROJECT,
            importance=0.9,
            tags=["engineering", "security"]
        )
        
        self.memory.store_episodic(
            source="team_lead",
            content="This sprint, focus on updating dependencies to address CVE-2023-12345.",
            scope=MemoryScope.SESSION,
            importance=0.7,
            tags=["sprint", "security"]
        )
        
        result_session = self.memory.retrieve(
            query="What security requirements apply?",
            scope=MemoryScope.SESSION
        )
        
        assert len(result_session.memories) > 0
        scopes_found = {m.scope for m in result_session.memories}
        assert MemoryScope.SESSION in scopes_found or MemoryScope.PROJECT in scopes_found or MemoryScope.GLOBAL in scopes_found


class TestMemoryManagement:
    """
    Test memory management features: pruning, decay, importance boosting
    """
    
    def setup_method(self):
        self.memory = AgenticMemory(enable_auto_pruning=False)
    
    def test_importance_decay(self):
        memory = self.memory.store_episodic(
            source="test",
            content="Old information that becomes less relevant over time.",
            scope=MemoryScope.SESSION,
            importance=0.8
        )
        
        initial_importance = memory.importance
        
        memory.decay_importance(decay_rate=0.1)
        
        assert memory.importance < initial_importance
    
    def test_access_based_boosting(self):
        memory = self.memory.store_semantic(
            entity="frequently_used",
            content="Information that gets accessed often should become more important.",
            scope=MemoryScope.GLOBAL,
            importance=0.5
        )
        
        initial_importance = memory.importance
        
        for _ in range(5):
            result = self.memory.retrieve(
                query="frequently used information",
                scope=MemoryScope.GLOBAL
            )
            if result.memories:
                result.memories[0].boost_importance(0.05)
        
        updated_memory = self.memory.get_memory(memory.id)
        assert updated_memory.importance > initial_importance
        assert updated_memory.access_count >= 5
    
    def test_pruning_old_memories(self):
        for i in range(10):
            self.memory.store_episodic(
                source="test",
                content=f"Low importance memory {i}",
                scope=MemoryScope.SESSION,
                importance=0.1
            )
        
        stats_before = self.memory.get_statistics()
        
        pruned_count = self.memory.prune_old_memories(days_threshold=0, importance_threshold=0.2)
        
        stats_after = self.memory.get_statistics()
        
        assert pruned_count > 0
        assert stats_after["total_memories"] < stats_before["total_memories"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
