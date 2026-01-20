from datetime import datetime
from typing import List, Dict
import math
from models.memory import Memory


class RelevanceScorer:
    def __init__(self, 
                 semantic_weight: float = 0.5,
                 recency_weight: float = 0.2,
                 importance_weight: float = 0.2,
                 access_weight: float = 0.1):
        self.semantic_weight = semantic_weight
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.access_weight = access_weight
    
    def score_memory(self, memory: Memory, semantic_similarity: float, 
                    current_time: datetime = None) -> float:
        if current_time is None:
            current_time = datetime.now()
        
        recency_score = self._calculate_recency(memory.timestamp, current_time)
        access_score = self._calculate_access_score(memory.access_count, memory.last_accessed, current_time)
        
        total_score = (
            self.semantic_weight * semantic_similarity +
            self.recency_weight * recency_score +
            self.importance_weight * memory.importance +
            self.access_weight * access_score
        ) * memory.confidence
        
        return total_score
    
    def _calculate_recency(self, timestamp: datetime, current_time: datetime) -> float:
        time_delta = (current_time - timestamp).total_seconds()
        hours_elapsed = time_delta / 3600
        
        decay_rate = 0.01
        recency_score = math.exp(-decay_rate * hours_elapsed)
        
        return recency_score
    
    def _calculate_access_score(self, access_count: int, last_accessed: datetime, 
                                current_time: datetime) -> float:
        if access_count == 0:
            return 0.0
        
        frequency_score = min(1.0, math.log(access_count + 1) / 5)
        
        if last_accessed:
            time_since_access = (current_time - last_accessed).total_seconds() / 3600
            recency_factor = math.exp(-0.05 * time_since_access)
        else:
            recency_factor = 0.0
        
        return frequency_score * 0.7 + recency_factor * 0.3
    
    def rank_memories(self, memories: List[Memory], similarities: List[float]) -> List[tuple]:
        scored_memories = []
        
        for memory, similarity in zip(memories, similarities):
            score = self.score_memory(memory, similarity)
            scored_memories.append((memory, score, similarity))
        
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return scored_memories


class ConflictDetector:
    def detect_contradictions(self, memories: List[Memory]) -> List[Dict]:
        conflicts = []
        
        for i, mem1 in enumerate(memories):
            for mem2 in memories[i+1:]:
                if self._are_contradictory(mem1, mem2):
                    conflicts.append({
                        "memory1": mem1,
                        "memory2": mem2,
                        "type": "contradiction"
                    })
        
        return conflicts
    
    def _are_contradictory(self, mem1: Memory, mem2: Memory) -> bool:
        if mem1.type != mem2.type:
            return False
        
        negation_pairs = [
            ("prefer", "dislike"),
            ("use", "avoid"),
            ("enable", "disable"),
            ("true", "false"),
            ("yes", "no")
        ]
        
        content1_lower = mem1.content.lower()
        content2_lower = mem2.content.lower()
        
        for pos, neg in negation_pairs:
            if (pos in content1_lower and neg in content2_lower) or \
               (neg in content1_lower and pos in content2_lower):
                return True
        
        return False
    
    def resolve_conflict(self, conflict: Dict, strategy: str = "recency") -> Memory:
        mem1 = conflict["memory1"]
        mem2 = conflict["memory2"]
        
        if strategy == "recency":
            return mem1 if mem1.timestamp > mem2.timestamp else mem2
        elif strategy == "confidence":
            return mem1 if mem1.confidence > mem2.confidence else mem2
        elif strategy == "importance":
            return mem1 if mem1.importance > mem2.importance else mem2
        else:
            return mem1
