from typing import List, Dict
import re
from models.memory import Memory, MemoryType


class ContextRefinement:
    def __init__(self, max_summary_ratio: float = 0.3):
        self.max_summary_ratio = max_summary_ratio
    
    def refine_memories(self, 
                       memories: List[Memory],
                       query: str,
                       max_tokens: int) -> List[Memory]:
        
        refined = []
        current_tokens = 0
        
        for memory in memories:
            if memory.type == MemoryType.EPISODIC and len(memory.content) > 500:
                processed = self._summarize_episodic(memory)
            elif memory.type == MemoryType.SEMANTIC:
                processed = self._prune_semantic(memory, query)
            else:
                processed = memory
            
            memory_tokens = self._estimate_tokens(processed.content)
            
            if current_tokens + memory_tokens <= max_tokens:
                refined.append(processed)
                current_tokens += memory_tokens
            else:
                break
        
        return refined
    
    def _summarize_episodic(self, memory: Memory) -> Memory:
        content = memory.content
        
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            return memory
        
        num_keep = max(3, int(len(sentences) * self.max_summary_ratio))
        
        important_sentences = sentences[:num_keep]
        
        summarized_content = ". ".join(important_sentences) + "."
        
        summarized_memory = memory.model_copy()
        summarized_memory.content = summarized_content
        summarized_memory.metadata["summarized"] = True
        summarized_memory.metadata["original_length"] = len(content)
        
        return summarized_memory
    
    def _prune_semantic(self, memory: Memory, query: str) -> Memory:
        content = memory.content
        
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        query_terms = set(query.lower().split())
        
        relevant_sentences = []
        for sentence in sentences:
            sentence_terms = set(sentence.lower().split())
            overlap = len(query_terms & sentence_terms)
            
            if overlap > 0 or len(relevant_sentences) == 0:
                relevant_sentences.append(sentence)
        
        if not relevant_sentences:
            return memory
        
        pruned_content = ". ".join(relevant_sentences) + "."
        
        pruned_memory = memory.model_copy()
        pruned_memory.content = pruned_content
        pruned_memory.metadata["pruned"] = True
        pruned_memory.metadata["original_sentences"] = len(sentences)
        pruned_memory.metadata["kept_sentences"] = len(relevant_sentences)
        
        return pruned_memory
    
    def _estimate_tokens(self, text: str) -> int:
        return len(text.split()) * 1.3
    
    def batch_summarize(self, memories: List[Memory]) -> str:
        summaries = []
        
        for memory in memories:
            summary = f"[{memory.type.value}] {memory.content[:100]}..."
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    def create_hierarchical_summary(self, memories: List[Memory]) -> Dict[str, str]:
        by_type = {}
        
        for memory in memories:
            if memory.type not in by_type:
                by_type[memory.type] = []
            by_type[memory.type].append(memory.content)
        
        summaries = {}
        for mem_type, contents in by_type.items():
            if len(contents) == 1:
                summaries[mem_type.value] = contents[0]
            else:
                summaries[mem_type.value] = f"{len(contents)} memories: " + "; ".join(
                    [c[:50] + "..." for c in contents[:3]]
                )
        
        return summaries
