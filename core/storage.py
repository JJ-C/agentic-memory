from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import json
from collections import defaultdict

from models.memory import Memory, MemoryType, MemoryScope
from utils.embeddings import EmbeddingManager


class MemoryStore:
    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        self.memories: Dict[str, Memory] = {}
        self.embeddings = embedding_manager or EmbeddingManager()
        
        self.scope_index: Dict[MemoryScope, List[str]] = defaultdict(list)
        self.type_index: Dict[MemoryType, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
    
    def store(self, 
              content: str,
              memory_type: MemoryType,
              scope: MemoryScope,
              metadata: Optional[Dict[str, Any]] = None,
              importance: float = 0.5,
              confidence: float = 1.0,
              tags: Optional[List[str]] = None) -> Memory:
        
        memory_id = str(uuid.uuid4())
        
        embedding = self.embeddings.embed(content)
        
        memory = Memory(
            id=memory_id,
            type=memory_type,
            scope=scope,
            content=content,
            metadata=metadata or {},
            importance=importance,
            confidence=confidence,
            tags=tags or [],
            embedding=embedding
        )
        
        self.memories[memory_id] = memory
        
        self.scope_index[scope].append(memory_id)
        self.type_index[memory_type].append(memory_id)
        for tag in memory.tags:
            self.tag_index[tag].append(memory_id)
        
        return memory
    
    def get(self, memory_id: str) -> Optional[Memory]:
        return self.memories.get(memory_id)
    
    def update(self, memory_id: str, **kwargs) -> Optional[Memory]:
        memory = self.memories.get(memory_id)
        if not memory:
            return None
        
        for key, value in kwargs.items():
            if hasattr(memory, key):
                setattr(memory, key, value)
        
        if "content" in kwargs:
            memory.embedding = self.embeddings.embed(kwargs["content"])
        
        return memory
    
    def delete(self, memory_id: str) -> bool:
        memory = self.memories.get(memory_id)
        if not memory:
            return False
        
        self.scope_index[memory.scope].remove(memory_id)
        self.type_index[memory.type].remove(memory_id)
        for tag in memory.tags:
            self.tag_index[tag].remove(memory_id)
        
        del self.memories[memory_id]
        return True
    
    def search_by_scope(self, scope: MemoryScope) -> List[Memory]:
        memory_ids = self.scope_index.get(scope, [])
        return [self.memories[mid] for mid in memory_ids]
    
    def search_by_type(self, memory_type: MemoryType) -> List[Memory]:
        memory_ids = self.type_index.get(memory_type, [])
        return [self.memories[mid] for mid in memory_ids]
    
    def search_by_tags(self, tags: List[str]) -> List[Memory]:
        memory_ids = set()
        for tag in tags:
            memory_ids.update(self.tag_index.get(tag, []))
        return [self.memories[mid] for mid in memory_ids]
    
    def semantic_search(self, 
                       query: str,
                       scope: Optional[MemoryScope] = None,
                       memory_types: Optional[List[MemoryType]] = None,
                       top_k: int = 10,
                       min_similarity: float = 0.0) -> List[tuple]:
        
        query_embedding = self.embeddings.embed(query)
        
        candidate_memories = list(self.memories.values())
        
        if scope:
            candidate_memories = [m for m in candidate_memories if m.scope == scope]
        
        if memory_types:
            candidate_memories = [m for m in candidate_memories if m.type in memory_types]
        
        if not candidate_memories:
            return []
        
        candidate_embeddings = [m.embedding for m in candidate_memories]
        similarities = self.embeddings.batch_similarity(query_embedding, candidate_embeddings)
        
        results = [
            (memory, similarity) 
            for memory, similarity in zip(candidate_memories, similarities)
            if similarity >= min_similarity
        ]
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def get_all_memories(self, 
                        scope: Optional[MemoryScope] = None,
                        memory_type: Optional[MemoryType] = None) -> List[Memory]:
        memories = list(self.memories.values())
        
        if scope:
            memories = [m for m in memories if m.scope == scope]
        if memory_type:
            memories = [m for m in memories if m.type == memory_type]
        
        return memories
    
    def prune_old_memories(self, days_threshold: int = 30, importance_threshold: float = 0.2):
        current_time = datetime.now()
        to_delete = []
        
        for memory_id, memory in self.memories.items():
            age_days = (current_time - memory.timestamp).days
            
            if age_days > days_threshold and memory.importance < importance_threshold:
                to_delete.append(memory_id)
        
        for memory_id in to_delete:
            self.delete(memory_id)
        
        return len(to_delete)
    
    def export_memories(self, filepath: str):
        data = {
            memory_id: memory.model_dump(mode='json')
            for memory_id, memory in self.memories.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def import_memories(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for memory_id, memory_data in data.items():
            memory = Memory(**memory_data)
            self.memories[memory_id] = memory
            
            self.scope_index[memory.scope].append(memory_id)
            self.type_index[memory.type].append(memory_id)
            for tag in memory.tags:
                self.tag_index[tag].append(memory_id)
