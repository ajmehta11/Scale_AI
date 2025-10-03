from typing import List, Dict, Tuple, Optional
import threading
import math
from model import Chunk


class KDNode:    
    def __init__(self, chunk_id: str, embedding: List[float], axis: int):
        self.chunk_id = chunk_id
        self.embedding = embedding
        self.axis = axis
        self.left: Optional[KDNode] = None
        self.right: Optional[KDNode] = None


class KDTreeIndex:
    
    def __init__(self, library_id: str):
        self.library_id: str = library_id
        self.root: Optional[KDNode] = None
        self.chunk_ids: List[str] = []
        self.embeddings_cache: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
        self._needs_rebuild = False
    
    def insert(self, chunk_id: str, embedding: List[float]) -> None:
        with self._lock:
            if chunk_id not in self.chunk_ids:
                self.chunk_ids.append(chunk_id)
                self.embeddings_cache[chunk_id] = list(embedding)
                self._needs_rebuild = True
    
    def delete(self, chunk_id: str) -> bool:
        with self._lock:
            if chunk_id in self.chunk_ids:
                self.chunk_ids.remove(chunk_id)
                self.embeddings_cache.pop(chunk_id, None)
                self._needs_rebuild = True
                return True
            return False
    
    def update(self, chunk_id: str, new_embedding: List[float]) -> bool:
        with self._lock:
            if chunk_id in self.embeddings_cache:
                self.embeddings_cache[chunk_id] = list(new_embedding)
                self._needs_rebuild = True
                return True
            return False
    
    def build(self) -> None:

        with self._lock:
            if not self.chunk_ids:
                self.root = None
                self._needs_rebuild = False
                return
            
            items = [(cid, self.embeddings_cache[cid]) for cid in self.chunk_ids]
            
            self.root = self._build_recursive(items, depth=0)
            self._needs_rebuild = False
    
    def _build_recursive(
        self,
        items: List[Tuple[str, List[float]]],
        depth: int
    ) -> Optional[KDNode]:
        
        if not items:
            return None
        
        dimensions = len(items[0][1])
        axis = depth % dimensions
        
        items.sort(key=lambda x: x[1][axis])
        
        median_idx = len(items) // 2
        chunk_id, embedding = items[median_idx]
        
        node = KDNode(chunk_id, embedding, axis)
        
        node.left = self._build_recursive(items[:median_idx], depth + 1)
        node.right = self._build_recursive(items[median_idx + 1:], depth + 1)
        
        return node
    
    def search(
        self,
        query_embedding: List[float],
        chunks_map: Dict[str, Chunk],
        k: int = 10,
        metadata_filter: Optional[Dict[str, any]] = None
    ) -> List[Tuple[str, float]]:

        with self._lock:
            if self._needs_rebuild:
                self.build()
            
            if self.root is None:
                return []
            
            best_results: List[Tuple[float, str]] = []
            
            self._search_recursive(
                node=self.root,
                query=query_embedding,
                chunks_map=chunks_map,
                metadata_filter=metadata_filter,
                best_results=best_results,
                k=k
            )
            
            results = [(chunk_id, sim) for sim, chunk_id in best_results]
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
    
    def _search_recursive(
        self,
        node: Optional[KDNode],
        query: List[float],
        chunks_map: Dict[str, Chunk],
        metadata_filter: Optional[Dict[str, any]],
        best_results: List[Tuple[float, str]],
        k: int
    ) -> None:

        if node is None:
            return
        
        chunk = chunks_map.get(node.chunk_id)
        if chunk and (not metadata_filter or self._matches_filter(chunk, metadata_filter)):
            similarity = self._cosine_similarity(query, node.embedding)
            
            if len(best_results) < k:
                best_results.append((similarity, node.chunk_id))
                best_results.sort(reverse=True) 
            elif similarity > best_results[-1][0]:
                best_results[-1] = (similarity, node.chunk_id)
                best_results.sort(reverse=True)
        
        axis = node.axis
        diff = query[axis] - node.embedding[axis]
        
        if diff < 0:
            closer_node = node.left
            further_node = node.right
        else:
            closer_node = node.right
            further_node = node.left
        
        self._search_recursive(
            closer_node, query, chunks_map, metadata_filter, best_results, k
        )
        

        if len(best_results) < k or self._should_search_other_side(query, node, best_results[-1][0]):
            self._search_recursive(
                further_node, query, chunks_map, metadata_filter, best_results, k
            )
    
    def _should_search_other_side(
        self,
        query: List[float],
        node: KDNode,
        worst_similarity: float
    ) -> bool:
        return True  
    
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
            if key not in chunk_metadata or chunk_metadata[key] != value:
                return False
        
        return True
    
    def get_size(self) -> int:
        with self._lock:
            return len(self.chunk_ids)
    
    def clear(self) -> None:
        with self._lock:
            self.chunk_ids.clear()
            self.embeddings_cache.clear()
            self.root = None
            self._needs_rebuild = False
    
    def needs_rebuild(self) -> bool:
        with self._lock:
            return self._needs_rebuild