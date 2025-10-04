from typing import List, Dict, Tuple, Optional
import threading
import math
from model import Chunk


class BruteForceIndex: 
    def __init__(self, library_id: str):
        self.library_id: str = library_id
        self.chunk_ids: List[str] = []
        self.embeddings_cache: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
    
    def insert(self, chunk_id: str, embedding: List[float]) -> None:
        with self._lock:
            if chunk_id not in self.chunk_ids:
                self.chunk_ids.append(chunk_id)
                self.embeddings_cache[chunk_id] = list(embedding)
    
    def delete(self, chunk_id: str) -> bool:
        with self._lock:
            if chunk_id in self.chunk_ids:
                self.chunk_ids.remove(chunk_id)
                self.embeddings_cache.pop(chunk_id, None)
                return True
            return False
    
    def update(self, chunk_id: str, new_embedding: List[float]) -> bool:
        with self._lock:
            if chunk_id in self.embeddings_cache:
                self.embeddings_cache[chunk_id] = list(new_embedding)
                return True
            return False
    
    def search(
        self,
        query_embedding: List[float],
        chunks_map: Dict[str, Chunk],
        k: int = 10,
        metadata_filter: Optional[Dict[str, any]] = None
    ) -> List[Tuple[str, float]]:
        with self._lock:
            if not self.chunk_ids:
                return []
            
            results: List[Tuple[str, float]] = []
            
            for chunk_id in self.chunk_ids:
                embedding = self.embeddings_cache.get(chunk_id)
                if embedding is None:
                    continue
                
                if metadata_filter:
                    chunk = chunks_map.get(chunk_id)
                    if chunk is None or not self._matches_filter(chunk, metadata_filter):
                        continue
                
                similarity = self._cosine_similarity(query_embedding, embedding)
                results.append((chunk_id, similarity))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if len(vec1) != len(vec2):
            raise ValueError(f"Vector dimensions must match: {len(vec1)} != {len(vec2)}")
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _matches_filter(self, chunk: Chunk, metadata_filter: Dict[str, any]) -> bool:
        chunk_metadata = chunk.get_metadata()
        
        for key, value in metadata_filter.items():
            if key not in chunk_metadata:
                return False
            
            if chunk_metadata[key] != value:
                return False
        
        return True
    
    def get_size(self) -> int:
        with self._lock:
            return len(self.chunk_ids)
    
    def clear(self) -> None:
        with self._lock:
            self.chunk_ids.clear()
            self.embeddings_cache.clear()