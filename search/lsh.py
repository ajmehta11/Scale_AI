from typing import List, Dict, Tuple, Optional, Set
import threading
import math
import random


class LSHIndex:
    def __init__(
        self,
        library_id: str,
        num_tables: int = 10,
        hash_size: int = 12,
        seed: int = 42
    ):

        self.library_id: str = library_id
        self.num_tables: int = num_tables
        self.hash_size: int = hash_size
        self.seed: int = seed
        
        self.hash_tables: List[Dict[str, Set[str]]] = [
            {} for _ in range(num_tables)
        ]
        
        self.hyperplanes: List[List[List[float]]] = []
        
        self.chunk_ids: List[str] = []
        self.embeddings_cache: Dict[str, List[float]] = {}
        
        self._lock = threading.RLock()
        
        self._hyperplanes_initialized = False
    
    def _initialize_hyperplanes(self, dimension: int) -> None:

        random.seed(self.seed)
        
        self.hyperplanes = []
        for _ in range(self.num_tables):
            table_hyperplanes = []
            for _ in range(self.hash_size):
                plane = [random.gauss(0, 1) for _ in range(dimension)]
                magnitude = math.sqrt(sum(x * x for x in plane))
                if magnitude > 0:
                    plane = [x / magnitude for x in plane]
                table_hyperplanes.append(plane)
            self.hyperplanes.append(table_hyperplanes)
        
        self._hyperplanes_initialized = True
    
    def _compute_hash(self, embedding: List[float], table_idx: int) -> str:
        hash_bits = []
        for hyperplane in self.hyperplanes[table_idx]:
            dot_product = sum(e * h for e, h in zip(embedding, hyperplane))
            hash_bits.append('1' if dot_product >= 0 else '0')
        
        return ''.join(hash_bits)
    
    def insert(self, chunk_id: str, embedding: List[float]) -> None:
        with self._lock:
            if not self._hyperplanes_initialized:
                self._initialize_hyperplanes(len(embedding))
            
            if chunk_id in self.embeddings_cache:
                return  
            
            self.chunk_ids.append(chunk_id)
            self.embeddings_cache[chunk_id] = list(embedding)
            
            for table_idx in range(self.num_tables):
                hash_value = self._compute_hash(embedding, table_idx)
                
                if hash_value not in self.hash_tables[table_idx]:
                    self.hash_tables[table_idx][hash_value] = set()
                self.hash_tables[table_idx][hash_value].add(chunk_id)
    
    def delete(self, chunk_id: str) -> bool:
        with self._lock:
            if chunk_id not in self.embeddings_cache:
                return False
            
            embedding = self.embeddings_cache[chunk_id]
            
            for table_idx in range(self.num_tables):
                hash_value = self._compute_hash(embedding, table_idx)
                if hash_value in self.hash_tables[table_idx]:
                    self.hash_tables[table_idx][hash_value].discard(chunk_id)
                    if not self.hash_tables[table_idx][hash_value]:
                        del self.hash_tables[table_idx][hash_value]
            
            self.chunk_ids.remove(chunk_id)
            del self.embeddings_cache[chunk_id]
            
            return True
    
    def update(self, chunk_id: str, new_embedding: List[float]) -> bool:
        with self._lock:
            if self.delete(chunk_id):
                self.insert(chunk_id, new_embedding)
                return True
            return False
    
    def search(
        self,
        query_embedding: List[float],
        chunks_map: Dict[str, 'Chunk'],
        k: int = 10,
        metadata_filter: Optional[Dict[str, any]] = None,
        num_candidates_multiplier: int = 3
    ) -> List[Tuple[str, float]]:

        with self._lock:
            if not self._hyperplanes_initialized or not self.chunk_ids:
                return []
            
            candidates: Set[str] = set()
            
            for table_idx in range(self.num_tables):
                hash_value = self._compute_hash(query_embedding, table_idx)
                
                if hash_value in self.hash_tables[table_idx]:
                    candidates.update(self.hash_tables[table_idx][hash_value])
            
            if not candidates:
                candidates = set(random.sample(
                    self.chunk_ids,
                    min(k * num_candidates_multiplier, len(self.chunk_ids))
                ))
            
            results: List[Tuple[str, float]] = []
            
            for chunk_id in candidates:
                chunk = chunks_map.get(chunk_id)
                
                if chunk is None:
                    continue
                
                if metadata_filter and not self._matches_filter(chunk, metadata_filter):
                    continue
                
                embedding = self.embeddings_cache[chunk_id]
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
    
    def _matches_filter(self, chunk: 'Chunk', metadata_filter: Dict[str, any]) -> bool:
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
            self.hash_tables = [
                {} for _ in range(self.num_tables)
            ]
            self.hyperplanes = []
            self._hyperplanes_initialized = False
    
    def get_statistics(self) -> Dict:
        with self._lock:
            total_buckets = sum(len(table) for table in self.hash_tables)
            
            bucket_sizes = []
            for table in self.hash_tables:
                for bucket in table.values():
                    bucket_sizes.append(len(bucket))
            
            avg_bucket_size = sum(bucket_sizes) / len(bucket_sizes) if bucket_sizes else 0
            max_bucket_size = max(bucket_sizes) if bucket_sizes else 0
            
            return {
                "num_chunks": len(self.chunk_ids),
                "num_tables": self.num_tables,
                "hash_size": self.hash_size,
                "total_buckets": total_buckets,
                "avg_bucket_size": avg_bucket_size,
                "max_bucket_size": max_bucket_size,
                "hyperplanes_initialized": self._hyperplanes_initialized
            }