from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(text, str):
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        else:
            embeddings = self.model.encode(text, convert_to_numpy=True)
            return embeddings.tolist()
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def batch_similarity(self, query_embedding: List[float], 
                        candidate_embeddings: List[List[float]]) -> List[float]:
        query_vec = np.array(query_embedding)
        candidate_vecs = np.array(candidate_embeddings)
        
        dot_products = np.dot(candidate_vecs, query_vec)
        query_norm = np.linalg.norm(query_vec)
        candidate_norms = np.linalg.norm(candidate_vecs, axis=1)
        
        similarities = dot_products / (query_norm * candidate_norms + 1e-10)
        return similarities.tolist()
