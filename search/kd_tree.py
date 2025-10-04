from typing import List, Dict, Tuple, Optional, Any
import threading
import heapq
import math
from model import Chunk


class KDNode:
    def __init__(self, chunk_id: str, embedding: List[float], axis: int):
        self.chunk_id = chunk_id
        self.embedding = embedding  
        self.axis = axis
        self.left: Optional["KDNode"] = None
        self.right: Optional["KDNode"] = None



class KDTreeIndex:


    def __init__(self, library_id: str):
        self.library_id: str = library_id
        self.root: Optional[KDNode] = None
        self.chunk_ids: List[str] = []
        self.embeddings_cache: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
        self._needs_rebuild = False

    def insert(self, chunk_id: str, embedding: List[float]) -> None:
        unit = self._normalize(embedding)
        if unit is None:
            raise ValueError("Embedding must have non-zero norm.")
        with self._lock:
            if chunk_id not in self.chunk_ids:
                self.chunk_ids.append(chunk_id)
                self.embeddings_cache[chunk_id] = unit
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
        unit = self._normalize(new_embedding)
        if unit is None:
            raise ValueError("Embedding must have non-zero norm.")
        with self._lock:
            if chunk_id in self.embeddings_cache:
                self.embeddings_cache[chunk_id] = unit
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

    def search(
        self,
        query_embedding: List[float],
        chunks_map: Dict[str, Chunk],
        k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:

        q = self._normalize(query_embedding)
        if q is None:
            return []

        with self._lock:
            if self._needs_rebuild:
                self.build()
            root = self.root

        if root is None or k <= 0:
            return []

        best_heap: List[Tuple[float, str]] = [] ## best_heap[0] is WORST of current top-k

        self._search_recursive(
            node=root,
            query=q,
            chunks_map=chunks_map,
            metadata_filter=metadata_filter,
            best_heap=best_heap,
            k=k,
        )

        best_heap.sort(key=lambda x: x[0], reverse=True)
        return [(cid, sim) for (sim, cid) in best_heap]

    def _build_recursive(
        self,
        items: List[Tuple[str, List[float]]],
        depth: int
    ) -> Optional[KDNode]:
        if not items:
            return None

        dims = len(items[0][1])
        axis = depth % dims
        items.sort(key=lambda x: x[1][axis])

        mid = len(items) // 2
        chunk_id, embedding = items[mid]
        node = KDNode(chunk_id, embedding, axis)

        node.left = self._build_recursive(items[:mid], depth + 1)
        node.right = self._build_recursive(items[mid + 1:], depth + 1)
        return node

    def _search_recursive(
        self,
        node: Optional[KDNode],
        query: List[float],
        chunks_map: Dict[str, Chunk],
        metadata_filter: Optional[Dict[str, Any]],
        best_heap: List[Tuple[float, str]],
        k: int
    ) -> None:
        if node is None:
            return

        ch = chunks_map.get(node.chunk_id)
        if ch is not None and (not metadata_filter or self._matches_filter(ch, metadata_filter)):
            sim = self._cosine_similarity_unit(query, node.embedding)
            if len(best_heap) < k:
                heapq.heappush(best_heap, (sim, node.chunk_id))
            elif sim > best_heap[0][0]:
                heapq.heapreplace(best_heap, (sim, node.chunk_id))

        axis = node.axis
        diff = query[axis] - node.embedding[axis]
        closer, further = (node.left, node.right) if diff < 0 else (node.right, node.left)

        self._search_recursive(closer, query, chunks_map, metadata_filter, best_heap, k)


        worst_sim = best_heap[0][0] if len(best_heap) == k else -1.0  
        if self._should_search_other_side(query, node, worst_sim, len(best_heap), k):
            self._search_recursive(further, query, chunks_map, metadata_filter, best_heap, k)

    def _should_search_other_side(
        self,
        query: List[float],
        node: KDNode,
        worst_similarity: float,
        heap_size: int,
        k: int
    ) -> bool:
        if heap_size < k:
            return True  # need more candidates

        s = max(min(worst_similarity, 1.0), -1.0)
        r = math.sqrt(max(0.0, 2.0 * (1.0 - s)))

        axis = node.axis
        slab_distance = abs(query[axis] - node.embedding[axis])
        return slab_distance <= r

    @staticmethod
    def _normalize(vec: List[float]) -> Optional[List[float]]:
        norm2 = sum(x * x for x in vec)
        if norm2 <= 0.0:
            return None
        inv = 1.0 / math.sqrt(norm2)
        return [x * inv for x in vec]

    @staticmethod
    def _cosine_similarity_unit(v1: List[float], v2: List[float]) -> float:
        # For unit vectors, cosine == dot product
        return sum(a * b for a, b in zip(v1, v2))

    @staticmethod
    def _matches_filter(chunk: Chunk, metadata_filter: Dict[str, Any]) -> bool:
        md = chunk.get_metadata()
        for k, v in metadata_filter.items():
            if k not in md or md[k] != v:
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