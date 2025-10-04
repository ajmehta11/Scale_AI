from typing import List, Dict, Tuple, Optional, Set, Any
import threading
import math
import random
import heapq
import numpy as np


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

        self.hash_tables: List[Dict[int, Set[str]]] = [{} for _ in range(num_tables)]

        self.hyperplanes: Optional[np.ndarray] = None

        self.dimension: Optional[int] = None
        self.chunk_ids: List[str] = []
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.chunk_hashes: Dict[str, List[int]] = {}

        self._lock = threading.RLock()
        self._rng = np.random.default_rng(self.seed)
        self._weights: Optional[np.ndarray] = None
        self._hyperplanes_initialized = False


    def _assert_dim(self, embedding: List[float]) -> None:
        if self.dimension is None:
            return
        if len(embedding) != self.dimension:
            raise ValueError(
                f"Vector dimensions must match index dimension {self.dimension}, "
                f"got {len(embedding)}."
            )

    def _initialize_hyperplanes(self, dimension: int) -> None:
        self.dimension = dimension
        hp = self._rng.normal(0.0, 1.0, size=(self.num_tables, self.hash_size, dimension))
        norms = np.linalg.norm(hp, axis=2, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.hyperplanes = hp / norms


        self._weights = (1 << np.arange(self.hash_size, dtype=np.uint64)).astype(np.uint64)

        self._hyperplanes_initialized = True

    def _normalize_vec(self, v: List[float]) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(arr)
        if n == 0.0:
            raise ValueError("Embedding must have non-zero norm.")
        return arr / n

    def _compute_hashes_all_tables_np(self, unit_embedding: np.ndarray) -> np.ndarray:
        dots = np.tensordot(self.hyperplanes, unit_embedding, axes=([2], [0]))  # (T, H)
        bits = (dots >= 0.0).astype(np.uint64)  # (T, H) in {0,1}
        keys = (bits * self._weights).sum(axis=1)  # (T,)
        return keys

    def _compute_hash_int(self, unit_embedding: np.ndarray, table_idx: int) -> int:
        dots = self.hyperplanes[table_idx] @ unit_embedding  
        bits = (dots >= 0.0).astype(np.uint64)
        key = int((bits * self._weights).sum())
        return key

    def _cosine_similarity_unit(self, unit_query: np.ndarray, unit_vec: np.ndarray) -> float:
        if unit_query.shape != unit_vec.shape:
            raise ValueError("Vector dimensions must match for cosine.")
        return float(unit_query @ unit_vec)

    def _matches_filter(self, chunk: "Chunk", metadata_filter: Dict[str, Any]) -> bool:
        md = chunk.get_metadata()
        for k, v in metadata_filter.items():
            if k not in md or md[k] != v:
                return False
        return True


    def insert(self, chunk_id: str, embedding: List[float]) -> None:
        with self._lock:
            if self.dimension is None:
                self._initialize_hyperplanes(len(embedding))
            self._assert_dim(embedding)

            if chunk_id in self.embeddings_cache:
                return  

            unit = self._normalize_vec(embedding)
            keys = self._compute_hashes_all_tables_np(unit)  # (T,)

            self.chunk_ids.append(chunk_id)
            self.embeddings_cache[chunk_id] = unit
            self.chunk_hashes[chunk_id] = [int(k) for k in keys.tolist()]

            for t, key in enumerate(self.chunk_hashes[chunk_id]):
                bucket = self.hash_tables[t].get(key)
                if bucket is None:
                    self.hash_tables[t][key] = {chunk_id}
                else:
                    bucket.add(chunk_id)

    def delete(self, chunk_id: str) -> bool:
        with self._lock:
            if chunk_id not in self.embeddings_cache:
                return False

            keys = self.chunk_hashes.get(chunk_id)
            if keys is not None:
                for t, key in enumerate(keys):
                    bucket = self.hash_tables[t].get(key)
                    if bucket is not None:
                        bucket.discard(chunk_id)
                        if not bucket:
                            del self.hash_tables[t][key]

            try:
                self.chunk_ids.remove(chunk_id)
            except ValueError:
                pass
            self.embeddings_cache.pop(chunk_id, None)
            self.chunk_hashes.pop(chunk_id, None)

            return True

    def update(self, chunk_id: str, new_embedding: List[float]) -> bool:
        with self._lock:
            if chunk_id not in self.embeddings_cache:
                return False
            self._assert_dim(new_embedding)
            self.delete(chunk_id)
            self.insert(chunk_id, new_embedding)
            return True

    def search(
        self,
        query_embedding: List[float],
        chunks_map: Dict[str, 'Chunk'],
        k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        num_candidates_multiplier: int = 3
    ) -> List[Tuple[str, float]]:
        with self._lock:
            if not self._hyperplanes_initialized or not self.chunk_ids:
                return []

            self._assert_dim(query_embedding)
            q_unit = self._normalize_vec(query_embedding)

            keys = self._compute_hashes_all_tables_np(q_unit)  # (T,)
            candidates: Set[str] = set()
            for t, key in enumerate(keys.tolist()):
                bucket = self.hash_tables[t].get(int(key))
                if bucket:
                    candidates.update(bucket)

            if not candidates:
                if self.chunk_ids:
                    sample_sz = min(k * max(1, num_candidates_multiplier), len(self.chunk_ids))
                    candidates = set(random.sample(self.chunk_ids, sample_sz))
                else:
                    return []

            results: List[Tuple[str, float]] = []
            for cid in candidates:
                chunk = chunks_map.get(cid)
                if chunk is None:
                    continue
                if metadata_filter and not self._matches_filter(chunk, metadata_filter):
                    continue

                vec = self.embeddings_cache.get(cid)
                if vec is None:
                    continue  # should not happen
                sim = self._cosine_similarity_unit(q_unit, vec)
                results.append((cid, sim))

            if not results:
                return []

            topk = heapq.nlargest(k, results, key=lambda x: x[1])
            return topk

    def get_size(self) -> int:
        with self._lock:
            return len(self.chunk_ids)

    def clear(self) -> None:
        with self._lock:
            self.chunk_ids.clear()
            self.embeddings_cache.clear()
            self.chunk_hashes.clear()
            self.hash_tables = [{} for _ in range(self.num_tables)]
            self.hyperplanes = None
            self._weights = None
            self.dimension = None
            self._hyperplanes_initialized = False

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            total_buckets = sum(len(table) for table in self.hash_tables)

            bucket_sizes: List[int] = []
            for table in self.hash_tables:
                for bucket in table.values():
                    bucket_sizes.append(len(bucket))

            avg_bucket_size = (sum(bucket_sizes) / len(bucket_sizes)) if bucket_sizes else 0.0
            max_bucket_size = max(bucket_sizes) if bucket_sizes else 0

            return {
                "num_chunks": len(self.chunk_ids),
                "num_tables": self.num_tables,
                "hash_size": self.hash_size,
                "dimension": self.dimension,
                "total_buckets": total_buckets,
                "avg_bucket_size": avg_bucket_size,
                "max_bucket_size": max_bucket_size,
                "hyperplanes_initialized": self._hyperplanes_initialized
            }