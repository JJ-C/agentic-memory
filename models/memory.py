from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryScope(str, Enum):
    GLOBAL = "global"
    PROJECT = "project"
    SESSION = "session"
    TASK = "task"


class Memory(BaseModel):
    id: str
    type: MemoryType
    scope: MemoryScope
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    access_count: int = Field(default=0)
    last_accessed: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    
    def update_access(self):
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def decay_importance(self, decay_rate: float = 0.01):
        time_delta = (datetime.now() - self.timestamp).days
        self.importance = max(0.0, self.importance - (decay_rate * time_delta))
    
    def boost_importance(self, boost: float = 0.1):
        self.importance = min(1.0, self.importance + boost)


class RetrievalQuery(BaseModel):
    query: str
    task_type: Optional[str] = None
    scope: Optional[MemoryScope] = None
    memory_types: Optional[List[MemoryType]] = None
    max_tokens: int = 2000
    top_k: int = 10
    min_relevance: float = 0.3
    include_metadata: bool = True


class RetrievalResult(BaseModel):
    memories: List[Memory]
    total_tokens: int
    relevance_scores: Dict[str, float]
    strategy_used: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
