from typing import List, Optional, Dict, Any
from datetime import datetime

from models.memory import Memory, MemoryType, MemoryScope, RetrievalQuery, RetrievalResult
from core.storage import MemoryStore
from core.quarantine import ContextQuarantine
from core.refinement import ContextRefinement
from utils.scoring import RelevanceScorer, ConflictDetector


class RetrievalOrchestrator:
    def __init__(self, 
                 memory_store: MemoryStore,
                 quarantine: Optional[ContextQuarantine] = None,
                 refinement: Optional[ContextRefinement] = None,
                 scorer: Optional[RelevanceScorer] = None):
        
        self.store = memory_store
        self.quarantine = quarantine or ContextQuarantine()
        self.refinement = refinement or ContextRefinement()
        self.scorer = scorer or RelevanceScorer()
        self.conflict_detector = ConflictDetector()
    
    def retrieve(self, query: RetrievalQuery, context_id: Optional[str] = None) -> RetrievalResult:
        memory_types = self._select_memory_types(query)
        
        scope = query.scope or MemoryScope.SESSION
        
        candidates = self._semantic_search(
            query.query,
            scope=scope,
            memory_types=memory_types,
            top_k=query.top_k * 5
        )
        
        quarantined = self.quarantine.filter_by_quarantine(
            [mem for mem, _ in candidates],
            current_scope=scope,
            context_id=context_id
        )
        
        candidate_tuples = [(mem, sim) for mem, sim in candidates if mem in quarantined]
        
        reranked = self._rerank_memories(candidate_tuples, query)
        
        top_memories = [mem for mem, _, _ in reranked[:query.top_k]]
        
        conflicts = self.conflict_detector.detect_contradictions(top_memories)
        resolved_memories = self._resolve_conflicts(top_memories, conflicts)
        
        refined_memories = self.refinement.refine_memories(
            resolved_memories,
            query.query,
            query.max_tokens
        )
        
        for memory in refined_memories:
            memory.update_access()
        
        total_tokens = sum(
            self.refinement._estimate_tokens(m.content) 
            for m in refined_memories
        )
        
        relevance_scores = {
            mem.id: score 
            for mem, score, _ in reranked[:len(refined_memories)]
        }
        
        return RetrievalResult(
            memories=refined_memories,
            total_tokens=int(total_tokens),
            relevance_scores=relevance_scores,
            strategy_used=self._get_strategy_name(query),
            metadata={
                "candidates_found": len(candidates),
                "after_quarantine": len(quarantined),
                "conflicts_detected": len(conflicts),
                "memory_types_used": [mt.value for mt in memory_types]
            }
        )
    
    def _select_memory_types(self, query: RetrievalQuery) -> List[MemoryType]:
        if query.memory_types:
            return query.memory_types
        
        task_type = query.task_type or "general"
        
        type_mapping = {
            "factual": [MemoryType.SEMANTIC],
            "recommendation": [MemoryType.SEMANTIC, MemoryType.PROCEDURAL],
            "conversational": [MemoryType.EPISODIC, MemoryType.SEMANTIC],
            "procedural": [MemoryType.PROCEDURAL, MemoryType.EPISODIC],
            "debugging": [MemoryType.EPISODIC, MemoryType.PROCEDURAL],
            "general": [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]
        }
        
        return type_mapping.get(task_type, type_mapping["general"])
    
    def _semantic_search(self,
                        query: str,
                        scope: MemoryScope,
                        memory_types: List[MemoryType],
                        top_k: int) -> List[tuple]:
        
        return self.store.semantic_search(
            query=query,
            scope=scope,
            memory_types=memory_types,
            top_k=top_k,
            min_similarity=0.1
        )
    
    def _rerank_memories(self, 
                        candidates: List[tuple],
                        query: RetrievalQuery) -> List[tuple]:
        
        memories = [mem for mem, _ in candidates]
        similarities = [sim for _, sim in candidates]
        
        scored = self.scorer.rank_memories(memories, similarities)
        
        filtered = [
            (mem, score, sim) 
            for mem, score, sim in scored 
            if sim >= query.min_relevance
        ]
        
        return filtered
    
    def _resolve_conflicts(self, 
                          memories: List[Memory],
                          conflicts: List[Dict]) -> List[Memory]:
        
        if not conflicts:
            return memories
        
        resolved = list(memories)
        memories_to_remove = set()
        
        for conflict in conflicts:
            winner = self.conflict_detector.resolve_conflict(conflict, strategy="recency")
            loser = conflict["memory1"] if winner == conflict["memory2"] else conflict["memory2"]
            
            if loser in resolved:
                memories_to_remove.add(loser.id)
        
        resolved = [m for m in resolved if m.id not in memories_to_remove]
        
        return resolved
    
    def _get_strategy_name(self, query: RetrievalQuery) -> str:
        components = []
        
        if query.memory_types:
            components.append("tool_loadout")
        else:
            components.append("dynamic_selection")
        
        components.append("semantic_search")
        components.append("reranking")
        
        if query.scope:
            components.append("quarantine")
        
        components.append("refinement")
        
        return "+".join(components)


class MemoryRecommender:
    def __init__(self):
        self.usage_stats: Dict[str, Dict[str, int]] = {}
    
    def recommend_memory_types(self, 
                               query: str,
                               task_type: Optional[str] = None) -> List[MemoryType]:
        
        query_lower = query.lower()
        
        keywords = {
            MemoryType.EPISODIC: ["conversation", "discussed", "mentioned", "said", "talked"],
            MemoryType.SEMANTIC: ["fact", "information", "prefer", "like", "use", "know"],
            MemoryType.PROCEDURAL: ["how to", "process", "workflow", "steps", "procedure"]
        }
        
        scores = {}
        for mem_type, kws in keywords.items():
            score = sum(1 for kw in kws if kw in query_lower)
            scores[mem_type] = score
        
        if task_type and task_type in self.usage_stats:
            for mem_type, count in self.usage_stats[task_type].items():
                if mem_type in scores:
                    scores[MemoryType(mem_type)] += count * 0.1
        
        if not any(scores.values()):
            return list(MemoryType)
        
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        recommended = [mem_type for mem_type, score in sorted_types if score > 0]
        
        if not recommended:
            recommended = [sorted_types[0][0]]
        
        return recommended[:2]
    
    def record_usage(self, task_type: str, memory_types: List[MemoryType]):
        if task_type not in self.usage_stats:
            self.usage_stats[task_type] = {}
        
        for mem_type in memory_types:
            key = mem_type.value
            self.usage_stats[task_type][key] = self.usage_stats[task_type].get(key, 0) + 1
