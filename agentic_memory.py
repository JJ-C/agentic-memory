from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

from models.memory import Memory, MemoryType, MemoryScope, RetrievalQuery, RetrievalResult
from core.storage import MemoryStore
from core.retrieval import RetrievalOrchestrator, MemoryRecommender
from core.quarantine import ContextQuarantine
from core.refinement import ContextRefinement
from utils.embeddings import EmbeddingManager
from utils.scoring import RelevanceScorer
from utils.task_classifier import TaskClassifier, ClassificationMethod


class AgenticMemory:
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 enable_auto_pruning: bool = True,
                 task_classification_method: Literal["keyword_matching", "auto_llm"] = "keyword_matching",
                 llm_client=None):
        """
        Initialize Agentic Memory Framework
        
        Args:
            embedding_model: Name of sentence-transformers model to use
            enable_auto_pruning: Automatically prune old memories
            task_classification_method: Method for automatic task type classification
                - "keyword_matching": Fast, rule-based classification (default)
                - "auto_llm": LLM-based intelligent classification (requires llm_client)
            llm_client: LLM client for auto_llm classification (required if using auto_llm)
        """
        self.embeddings = EmbeddingManager(embedding_model)
        self.store = MemoryStore(self.embeddings)
        self.quarantine = ContextQuarantine()
        self.refinement = ContextRefinement()
        self.scorer = RelevanceScorer()
        self.orchestrator = RetrievalOrchestrator(
            self.store,
            self.quarantine,
            self.refinement,
            self.scorer
        )
        self.recommender = MemoryRecommender()
        
        # Task classifier for automatic task type detection
        self.task_classifier = TaskClassifier(
            method=ClassificationMethod(task_classification_method),
            llm_client=llm_client
        )
        
        self.enable_auto_pruning = enable_auto_pruning
        self.operation_count = 0
    
    def store_episodic(self,
                      source: str,
                      content: str,
                      scope: MemoryScope = MemoryScope.SESSION,
                      metadata: Optional[Dict[str, Any]] = None,
                      importance: float = 0.5,
                      tags: Optional[List[str]] = None) -> Memory:
        
        if metadata is None:
            metadata = {}
        metadata["source"] = source
        
        return self.store.store(
            content=content,
            memory_type=MemoryType.EPISODIC,
            scope=scope,
            metadata=metadata,
            importance=importance,
            tags=tags or []
        )
    
    def store_semantic(self,
                      entity: str,
                      content: str,
                      scope: MemoryScope = MemoryScope.GLOBAL,
                      metadata: Optional[Dict[str, Any]] = None,
                      importance: float = 0.7,
                      confidence: float = 1.0,
                      tags: Optional[List[str]] = None) -> Memory:
        
        if metadata is None:
            metadata = {}
        metadata["entity"] = entity
        
        return self.store.store(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            scope=scope,
            metadata=metadata,
            importance=importance,
            confidence=confidence,
            tags=tags or []
        )
    
    def store_procedural(self,
                        workflow_name: str,
                        content: str,
                        scope: MemoryScope = MemoryScope.PROJECT,
                        metadata: Optional[Dict[str, Any]] = None,
                        importance: float = 0.6,
                        tags: Optional[List[str]] = None) -> Memory:
        
        if metadata is None:
            metadata = {}
        metadata["workflow"] = workflow_name
        
        return self.store.store(
            content=content,
            memory_type=MemoryType.PROCEDURAL,
            scope=scope,
            metadata=metadata,
            importance=importance,
            tags=tags or []
        )
    
    def retrieve(self,
                query: str,
                task_type: Optional[str] = None,
                scope: Optional[MemoryScope] = None,
                memory_types: Optional[List[MemoryType]] = None,
                max_tokens: int = 2000,
                top_k: int = 10,
                context_id: Optional[str] = None,
                auto_classify: bool = True) -> RetrievalResult:
        """
        Retrieve relevant memories for a query
        
        Args:
            query: Search query
            task_type: Task type (factual, recommendation, conversational, procedural, debugging, general)
                      If None and auto_classify=True, will be automatically determined
            scope: Memory scope to search within
            memory_types: Specific memory types to search
            max_tokens: Maximum tokens in result
            top_k: Maximum number of memories to retrieve
            context_id: Context identifier for isolation
            auto_classify: Automatically classify task type if not provided (default: True)
        
        Returns:
            RetrievalResult with memories and metadata
        """
        # Automatically classify task type if not provided
        if task_type is None and auto_classify:
            task_type = self.task_classifier.classify(query)
        
        retrieval_query = RetrievalQuery(
            query=query,
            task_type=task_type,
            scope=scope,
            memory_types=memory_types,
            max_tokens=max_tokens,
            top_k=top_k
        )
        
        result = self.orchestrator.retrieve(retrieval_query, context_id)
        
        # Add classification info to metadata
        if auto_classify and task_type:
            result.metadata["auto_classified_task_type"] = task_type
        
        if task_type:
            used_types = [m.type for m in result.memories]
            self.recommender.record_usage(task_type, used_types)
        
        self._maybe_auto_prune()
        
        return result
    
    def classify_query(self, query: str) -> str:
        """
        Classify a query into a task type
        
        Args:
            query: Query to classify
            
        Returns:
            Task type string (factual, recommendation, conversational, procedural, debugging, general)
        """
        return self.task_classifier.classify(query)
    
    def update_memory(self, memory_id: str, **kwargs) -> Optional[Memory]:
        return self.store.update(memory_id, **kwargs)
    
    def delete_memory(self, memory_id: str) -> bool:
        return self.store.delete(memory_id)
    
    def get_memory(self, memory_id: str) -> Optional[Memory]:
        return self.store.get(memory_id)
    
    def create_isolated_context(self, context_id: str, scope: MemoryScope):
        self.quarantine.create_isolated_context(context_id, scope)
    
    def grant_context_access(self, from_context: str, to_context: str):
        self.quarantine.grant_cross_boundary_access(from_context, to_context)
    
    def clear_context(self, context_id: str):
        self.quarantine.clear_context(context_id)
    
    def prune_old_memories(self, days_threshold: int = 30, importance_threshold: float = 0.2) -> int:
        return self.store.prune_old_memories(days_threshold, importance_threshold)
    
    def get_statistics(self) -> Dict[str, Any]:
        all_memories = self.store.get_all_memories()
        
        by_type = {}
        by_scope = {}
        
        for memory in all_memories:
            by_type[memory.type.value] = by_type.get(memory.type.value, 0) + 1
            by_scope[memory.scope.value] = by_scope.get(memory.scope.value, 0) + 1
        
        return {
            "total_memories": len(all_memories),
            "by_type": by_type,
            "by_scope": by_scope,
            "operation_count": self.operation_count
        }
    
    def export_to_file(self, filepath: str):
        self.store.export_memories(filepath)
    
    def import_from_file(self, filepath: str):
        self.store.import_memories(filepath)
    
    def _maybe_auto_prune(self):
        self.operation_count += 1
        
        if self.enable_auto_pruning and self.operation_count % 100 == 0:
            pruned = self.prune_old_memories()
            if pruned > 0:
                print(f"Auto-pruned {pruned} old memories")
