from typing import List, Optional, Dict, Set
from models.memory import Memory, MemoryScope


class ContextQuarantine:
    def __init__(self):
        self.scope_hierarchy = {
            MemoryScope.GLOBAL: [],
            MemoryScope.PROJECT: [MemoryScope.GLOBAL],
            MemoryScope.SESSION: [MemoryScope.GLOBAL, MemoryScope.PROJECT],
            MemoryScope.TASK: [MemoryScope.GLOBAL, MemoryScope.PROJECT, MemoryScope.SESSION]
        }
        
        self.active_contexts: Dict[str, Set[MemoryScope]] = {}
        
        self.cross_boundary_permissions: Dict[str, Set[str]] = {}
    
    def get_accessible_scopes(self, current_scope: MemoryScope) -> List[MemoryScope]:
        accessible = [current_scope]
        accessible.extend(self.scope_hierarchy.get(current_scope, []))
        return accessible
    
    def filter_by_quarantine(self, 
                            memories: List[Memory],
                            current_scope: MemoryScope,
                            context_id: Optional[str] = None) -> List[Memory]:
        
        accessible_scopes = set(self.get_accessible_scopes(current_scope))
        
        if context_id and context_id in self.cross_boundary_permissions:
            for permitted_context in self.cross_boundary_permissions[context_id]:
                accessible_scopes.update(self.get_accessible_scopes(MemoryScope.SESSION))
        
        filtered = [m for m in memories if m.scope in accessible_scopes]
        
        return filtered
    
    def create_isolated_context(self, context_id: str, scope: MemoryScope):
        self.active_contexts[context_id] = {scope}
    
    def grant_cross_boundary_access(self, from_context: str, to_context: str):
        if from_context not in self.cross_boundary_permissions:
            self.cross_boundary_permissions[from_context] = set()
        self.cross_boundary_permissions[from_context].add(to_context)
    
    def revoke_cross_boundary_access(self, from_context: str, to_context: str):
        if from_context in self.cross_boundary_permissions:
            self.cross_boundary_permissions[from_context].discard(to_context)
    
    def clear_context(self, context_id: str):
        if context_id in self.active_contexts:
            del self.active_contexts[context_id]
        if context_id in self.cross_boundary_permissions:
            del self.cross_boundary_permissions[context_id]
    
    def is_accessible(self, memory: Memory, current_scope: MemoryScope, 
                     context_id: Optional[str] = None) -> bool:
        accessible_scopes = self.get_accessible_scopes(current_scope)
        
        if memory.scope in accessible_scopes:
            return True
        
        if context_id and context_id in self.cross_boundary_permissions:
            return True
        
        return False
