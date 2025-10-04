import numpy as np
import heapq
import threading
from typing import List, Dict, Tuple, Optional, Set

class IVFPQIndex:
    def __init__(self, library_id: str, num_clusters: int = 100, num_subspaces: int = 8,
                 num_pq_clusters: int = 256, training_iterations: int = 10,
                 nprobe: int = 10, seed: int = 42):
        self.library_id = library_id
        self.num_clusters = num_clusters
        self.num_subspaces = num_subspaces
        self.num_pq_clusters = num_pq_clusters
        self.training_iterations = training_iterations
        self.nprobe = nprobe
        self.seed = seed
        
        self.dimension: Optional[int] = None
        self.subspace_dim: int = 0
        self._trained: bool = False
        self._lock = threading.RLock()
        
        self.coarse_centroids: List[List[float]] = []
        self.pq_codebooks: List[List[List[float]]] = []
        self.embeddings_cache: Dict[str, List[float]] = {}
        
        self._codes_np: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._id_to_row: Dict[str, int] = {}
        self._row_cluster: List[int] = []
        self._row_pos_in_cluster: List[int] = []
        self.inverted_lists: List[List[int]] = [[] for _ in range(num_clusters)]

    def train(self, embeddings: List[List[float]]) -> None:
        with self._lock:
            if self._trained:
                return
            
            if not embeddings:
                return
            
            self.dimension = len(embeddings[0])
            self.subspace_dim = self.dimension // self.num_subspaces
            
            X = np.array(embeddings, dtype=np.float32)
            
            self.coarse_centroids = self._kmeans(X, min(self.num_clusters, len(X)))
            
            self.pq_codebooks = []
            for m in range(self.num_subspaces):
                start_idx = m * self.subspace_dim
                end_idx = start_idx + self.subspace_dim
                subvector = X[:, start_idx:end_idx]
                
                codebook = self._kmeans(subvector, min(self.num_pq_clusters, len(subvector)))
                self.pq_codebooks.append(codebook)
            
            self._trained = True

    def _kmeans(self, X: np.ndarray, k: int, max_iter: int = 10) -> List[List[float]]:
        if len(X) == 0:
            return []
        
        if len(X) <= k:
            return X.tolist()
        
        np.random.seed(self.seed)
        indices = np.random.choice(len(X), k, replace=False)
        centroids = X[indices].copy()
        
        for _ in range(max_iter):
            distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            
            new_centroids = centroids.copy()
            for i in range(k):
                mask = labels == i
                if np.sum(mask) > 0:
                    new_centroids[i] = np.mean(X[mask], axis=0)
            
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        return centroids.tolist()

    def _find_nearest_cluster(self, embedding: List[float]) -> int:
        min_dist = float('inf')
        nearest = 0
        
        for i, centroid in enumerate(self.coarse_centroids):
            dist = self._euclidean_distance(embedding, centroid)
            if dist < min_dist:
                min_dist = dist
                nearest = i
        
        return nearest

    def _euclidean_distance(self, v1: List[float], v2: List[float]) -> float:
        return sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5

    def _subtract_vectors(self, v1: List[float], v2: List[float]) -> List[float]:
        return [a - b for a, b in zip(v1, v2)]

    def _encode_residual(self, residual: List[float]) -> List[int]:
        codes = []
        
        for m in range(self.num_subspaces):
            start_idx = m * self.subspace_dim
            end_idx = start_idx + self.subspace_dim
            subvector = residual[start_idx:end_idx]
            
            min_dist = float('inf')
            nearest = 0
            
            for i, centroid in enumerate(self.pq_codebooks[m]):
                dist = sum((a - b) ** 2 for a, b in zip(subvector, centroid))
                if dist < min_dist:
                    min_dist = dist
                    nearest = i
            
            codes.append(nearest)
        
        return codes

    def insert(self, chunk_id: str, embedding: List[float]) -> None:
        with self._lock:
            if chunk_id in self._id_to_row:
                return
            if not self._trained:
                self.train([embedding])

            cluster_id = self._find_nearest_cluster(embedding)

            residual = self._subtract_vectors(embedding, self.coarse_centroids[cluster_id])
            pq_codes = self._encode_residual(residual)

            code_row = np.array(pq_codes, dtype=np.uint8)[None, :]
            if self._codes_np is None:
                self._codes_np = code_row
            else:
                self._codes_np = np.vstack((self._codes_np, code_row))

            row = len(self._ids)
            self._ids.append(chunk_id)
            self._id_to_row[chunk_id] = row
            self._row_cluster.append(cluster_id)

            pos = len(self.inverted_lists[cluster_id])
            self.inverted_lists[cluster_id].append(row)
            self._row_pos_in_cluster.append(pos)

            self.embeddings_cache[chunk_id] = list(embedding)

    def delete(self, chunk_id: str) -> bool:
        with self._lock:
            if chunk_id not in self._id_to_row:
                return False

            row = self._id_to_row.pop(chunk_id)
            last_row = len(self._ids) - 1
            cluster_id = self._row_cluster[row]
            pos = self._row_pos_in_cluster[row]

            clist = self.inverted_lists[cluster_id]
            last_row_in_clist = clist[-1]
            clist[pos] = last_row_in_clist
            clist.pop()
            self._row_pos_in_cluster[last_row_in_clist] = pos

            if row != last_row:
                self._codes_np[row, :] = self._codes_np[last_row, :]
                moved_id = self._ids[last_row]
                self._ids[row] = moved_id
                self._id_to_row[moved_id] = row

                self._row_cluster[row] = self._row_cluster[last_row]
                self._row_pos_in_cluster[row] = self._row_pos_in_cluster[last_row]

                moved_cluster = self._row_cluster[row]
                moved_pos = self._row_pos_in_cluster[row]
                self.inverted_lists[moved_cluster][moved_pos] = row

            self._codes_np = self._codes_np[:-1, :] if self._codes_np is not None else None
            self._ids.pop()
            self._row_cluster.pop()
            self._row_pos_in_cluster.pop()

            if chunk_id in self.embeddings_cache:
                del self.embeddings_cache[chunk_id]

            return True

    def update(self, chunk_id: str, new_embedding: List[float]) -> bool:
        with self._lock:
            if chunk_id not in self._id_to_row:
                return False
            self.delete(chunk_id)
            self.insert(chunk_id, new_embedding)
            return True

    def search(
        self,
        query_embedding: List[float],
        chunks_map: Dict[str, 'Chunk'],
        k: int = 10,
        metadata_filter: Optional[Dict[str, any]] = None,
        refine_factor: int = 3
    ) -> List[Tuple[str, float]]:

        with self._lock:
            if not self._trained or self._codes_np is None or self._codes_np.shape[0] == 0:
                return []

            D = self.dimension
            M = self.num_subspaces
            Ks = self.num_pq_clusters
            q = np.asarray(query_embedding, dtype=np.float32)

            cluster_distances = []
            for c_idx, c in enumerate(self.coarse_centroids):
                dist = self._euclidean_distance(query_embedding, c)
                cluster_distances.append((c_idx, dist))
            cluster_distances.sort(key=lambda x: x[1])
            probe_clusters = [c for c, _ in cluster_distances[:self.nprobe]]

            approx_heap: List[Tuple[float, int]] = []
            target_candidates = k * max(1, refine_factor)

            subdim = self.subspace_dim

            def split_into_subvectors(vec: np.ndarray) -> np.ndarray:
                return vec.reshape(M, subdim)

            for c_idx in probe_clusters:
                rows = self.inverted_lists[c_idx]
                if not rows:
                    continue

                c = np.asarray(self.coarse_centroids[c_idx], dtype=np.float32)
                q_res = q - c
                q_res_blocks = split_into_subvectors(q_res)

                LUT = np.empty((M, Ks), dtype=np.float32)
                for m in range(M):
                    cb = np.asarray(self.pq_codebooks[m], dtype=np.float32)
                    diff = cb - q_res_blocks[m][None, :]
                    LUT[m, :] = np.sum(diff * diff, axis=1)

                codes = self._codes_np[rows, :]
                codes_T = codes.T
                adc_mr = LUT[np.arange(M)[:, None], codes_T]
                adc_rows = np.sum(adc_mr, axis=0)

                for row, dist_sq in zip(rows, adc_rows):
                    score = -float(dist_sq)
                    if len(approx_heap) < target_candidates:
                        heapq.heappush(approx_heap, (score, row))
                    else:
                        if score > approx_heap[0][0]:
                            heapq.heapreplace(approx_heap, (score, row))

            if not approx_heap:
                return []

            top_approx = heapq.nlargest(min(len(approx_heap), target_candidates), approx_heap)
            candidate_rows = [row for (_score, row) in top_approx]

            if metadata_filter:
                filtered_rows = []
                for r in candidate_rows:
                    chunk_id = self._ids[r]
                    chunk = chunks_map.get(chunk_id)
                    if chunk is not None:
                        ok = True
                        md = getattr(chunk, 'get_metadata', None)
                        md = md() if callable(md) else getattr(chunk, 'metadata', {})
                        for kf, vf in metadata_filter.items():
                            if md.get(kf) != vf:
                                ok = False
                                break
                        if ok:
                            filtered_rows.append(r)
                candidate_rows = filtered_rows
                if not candidate_rows:
                    return []

            E = np.empty((len(candidate_rows), D), dtype=np.float32)
            ids = []
            for i, r in enumerate(candidate_rows):
                cid = self._ids[r]
                ids.append(cid)
                E[i, :] = np.asarray(self.embeddings_cache[cid], dtype=np.float32)

            q_norm = np.linalg.norm(q)
            if q_norm == 0:
                return []
            dots = E @ q
            e_norms = np.linalg.norm(E, axis=1)
            denom = e_norms * q_norm
            denom[denom == 0] = 1e-12
            sims = dots / denom

            topk_idx = np.argpartition(-sims, kth=min(k, len(sims)-1))[:k]
            topk_sorted = topk_idx[np.argsort(-sims[topk_idx])]

            return [(ids[i], float(sims[i])) for i in topk_sorted]

    def clear(self) -> None:
        with self._lock:
            self._codes_np = None
            self._ids = []
            self._id_to_row = {}
            self._row_cluster = []
            self._row_pos_in_cluster = []
            self.inverted_lists = [[] for _ in range(self.num_clusters)]
            self.embeddings_cache = {}
            self._trained = False
            self.coarse_centroids = []
            self.pq_codebooks = []