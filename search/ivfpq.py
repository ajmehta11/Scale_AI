from typing import List, Dict, Tuple, Optional, Set
import threading
import math
import random


class IVFPQIndex:
    
    def __init__(
        self,
        library_id: str,
        num_clusters: int = 100,
        num_subspaces: int = 8,
        num_pq_clusters: int = 256,
        training_iterations: int = 10,
        nprobe: int = 10,
        seed: int = 42
    ):

        self.library_id: str = library_id
        self.num_clusters: int = num_clusters
        self.num_subspaces: int = num_subspaces
        self.num_pq_clusters: int = num_pq_clusters
        self.training_iterations: int = training_iterations
        self.nprobe: int = nprobe
        self.seed: int = seed
        
        self.coarse_centroids: List[List[float]] = []
        
        self.inverted_lists: List[List[str]] = []
        
        self.pq_codebooks: List[List[List[float]]] = []
        
        self.compressed_data: Dict[str, Tuple[int, List[int]]] = {}
        
        self.embeddings_cache: Dict[str, List[float]] = {}
        
        self.chunk_ids: List[str] = []
        
        self.dimension: Optional[int] = None
        self.subspace_dim: Optional[int] = None
        
        self._trained = False
        self._lock = threading.RLock()
    
    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
    
    def _subtract_vectors(self, vec1: List[float], vec2: List[float]) -> List[float]:
        return [a - b for a, b in zip(vec1, vec2)]
    
    def _add_vectors(self, vec1: List[float], vec2: List[float]) -> List[float]:
        return [a + b for a, b in zip(vec1, vec2)]
    
    def _kmeans_clustering(
        self,
        vectors: List[List[float]],
        k: int
    ) -> List[List[float]]:

        if not vectors:
            return []
        
        n = len(vectors)
        dim = len(vectors[0])
        
        random.seed(self.seed)
        centroids = random.sample(vectors, min(k, n))
        
        while len(centroids) < k:
            centroids.append([random.gauss(0, 1) for _ in range(dim)])
        
        for iteration in range(self.training_iterations):
            assignments = [0] * n
            for i, vec in enumerate(vectors):
                min_dist = float('inf')
                best_cluster = 0
                for c_idx, centroid in enumerate(centroids):
                    dist = self._euclidean_distance(vec, centroid)
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = c_idx
                assignments[i] = best_cluster
            
            new_centroids = []
            for c_idx in range(k):
                cluster_vecs = [vectors[i] for i in range(n) if assignments[i] == c_idx]
                if cluster_vecs:
                    centroid = [
                        sum(vec[d] for vec in cluster_vecs) / len(cluster_vecs)
                        for d in range(dim)
                    ]
                    new_centroids.append(centroid)
                else:
                    new_centroids.append(centroids[c_idx])
            
            centroids = new_centroids
        
        return centroids
    
    def _split_vector(self, vector: List[float]) -> List[List[float]]:
        subvectors = []
        for m in range(self.num_subspaces):
            start_idx = m * self.subspace_dim
            end_idx = start_idx + self.subspace_dim
            subvectors.append(vector[start_idx:end_idx])
        return subvectors
    
    def train(self, training_embeddings: List[List[float]]) -> None:
 
        with self._lock:
            if not training_embeddings:
                raise ValueError("Need training data")
            
            self.dimension = len(training_embeddings[0])
            
            if self.dimension % self.num_subspaces != 0:
                raise ValueError(
                    f"Dimension {self.dimension} must be divisible by "
                    f"num_subspaces {self.num_subspaces}"
                )
            
            self.subspace_dim = self.dimension // self.num_subspaces
            
            print(f"Training IVFPQ: {self.num_clusters} clusters, "
                  f"{self.num_subspaces} PQ subspaces")
            
            print("  Stage 1: Training coarse quantizer...")
            self.coarse_centroids = self._kmeans_clustering(
                training_embeddings,
                self.num_clusters
            )
            
            self.inverted_lists = [[] for _ in range(self.num_clusters)]
            print(f"  ✓ Trained {len(self.coarse_centroids)} coarse clusters")
            
            print("  Stage 2: Training PQ on residuals...")
            
            residuals = []
            for vec in training_embeddings:
                min_dist = float('inf')
                best_cluster = 0
                for c_idx, centroid in enumerate(self.coarse_centroids):
                    dist = self._euclidean_distance(vec, centroid)
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = c_idx
                
                residual = self._subtract_vectors(vec, self.coarse_centroids[best_cluster])
                residuals.append(residual)
            
            self.pq_codebooks = []
            for m in range(self.num_subspaces):
                subvectors = [self._split_vector(res)[m] for res in residuals]
                
                centroids = self._kmeans_clustering(subvectors, self.num_pq_clusters)
                self.pq_codebooks.append(centroids)
                
                if (m + 1) % 2 == 0:
                    print(f"    Trained PQ codebook {m + 1}/{self.num_subspaces}")
            
            self._trained = True
            print("  ✓ Training complete")
    
    def _encode_residual(self, residual: List[float]) -> List[int]:
        subvectors = self._split_vector(residual)
        codes = []
        
        for m, subvec in enumerate(subvectors):
            min_dist = float('inf')
            best_code = 0
            
            for c_idx, centroid in enumerate(self.pq_codebooks[m]):
                dist = self._euclidean_distance(subvec, centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_code = c_idx
            
            codes.append(best_code)
        
        return codes
    
    def _find_nearest_cluster(self, vector: List[float]) -> int:
        min_dist = float('inf')
        best_cluster = 0
        
        for c_idx, centroid in enumerate(self.coarse_centroids):
            dist = self._euclidean_distance(vector, centroid)
            if dist < min_dist:
                min_dist = dist
                best_cluster = c_idx
        
        return best_cluster
    
    def insert(self, chunk_id: str, embedding: List[float]) -> None:

        with self._lock:
            if chunk_id in self.compressed_data:
                return  
            
            if not self._trained:
                print("Auto-training on first insertion...")
                self.train([embedding])
            
            cluster_id = self._find_nearest_cluster(embedding)
            
            residual = self._subtract_vectors(embedding, self.coarse_centroids[cluster_id])
            
            pq_codes = self._encode_residual(residual)
            
            self.compressed_data[chunk_id] = (cluster_id, pq_codes)
            self.embeddings_cache[chunk_id] = list(embedding)
            self.inverted_lists[cluster_id].append(chunk_id)
            self.chunk_ids.append(chunk_id)
    
    def delete(self, chunk_id: str) -> bool:

        with self._lock:
            if chunk_id not in self.compressed_data:
                return False
            
            # Get cluster info
            cluster_id, _ = self.compressed_data[chunk_id]
            
            # Remove from inverted list
            self.inverted_lists[cluster_id].remove(chunk_id)
            
            # Remove from storage
            del self.compressed_data[chunk_id]
            del self.embeddings_cache[chunk_id]
            self.chunk_ids.remove(chunk_id)
            
            return True
    
    def update(self, chunk_id: str, new_embedding: List[float]) -> bool:
      
        with self._lock:
            if self.delete(chunk_id):
                self.insert(chunk_id, new_embedding)
                return True
            return False
    
    def _asymmetric_distance_residual(
        self,
        query_residual: List[float],
        pq_codes: List[int]
    ) -> float:

        query_subvectors = self._split_vector(query_residual)
        distance = 0.0
        
        for m, (query_subvec, code) in enumerate(zip(query_subvectors, pq_codes)):
            centroid = self.pq_codebooks[m][code]
            dist = sum((q - c) ** 2 for q, c in zip(query_subvec, centroid))
            distance += dist
        
        return math.sqrt(distance)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if len(vec1) != len(vec2):
            raise ValueError("Vector dimensions must match")
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def search(
        self,
        query_embedding: List[float],
        chunks_map: Dict[str, 'Chunk'],
        k: int = 10,
        metadata_filter: Optional[Dict[str, any]] = None,
        refine_factor: int = 3
    ) -> List[Tuple[str, float]]:

        with self._lock:
            if not self._trained or not self.chunk_ids:
                return []
            
            cluster_distances = []
            for c_idx, centroid in enumerate(self.coarse_centroids):
                dist = self._euclidean_distance(query_embedding, centroid)
                cluster_distances.append((c_idx, dist))
            
            cluster_distances.sort(key=lambda x: x[1])
            probe_clusters = [c_idx for c_idx, _ in cluster_distances[:self.nprobe]]
            
            candidates = []
            
            for cluster_id in probe_clusters:
                query_residual = self._subtract_vectors(
                    query_embedding,
                    self.coarse_centroids[cluster_id]
                )
                
                for chunk_id in self.inverted_lists[cluster_id]:
                    chunk = chunks_map.get(chunk_id)
                    if chunk is None:
                        continue
                    
                    if metadata_filter and not self._matches_filter(chunk, metadata_filter):
                        continue
                    
                    stored_cluster_id, pq_codes = self.compressed_data[chunk_id]
                    
                    approx_dist = self._asymmetric_distance_residual(
                        query_residual,
                        pq_codes
                    )
                    
                    candidates.append((chunk_id, -approx_dist))
            
            if not candidates:
                return []
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            num_candidates = min(k * refine_factor, len(candidates))
            top_candidates = candidates[:num_candidates]
            
            refined_results = []
            for chunk_id, _ in top_candidates:
                exact_embedding = self.embeddings_cache[chunk_id]
                similarity = self._cosine_similarity(query_embedding, exact_embedding)
                refined_results.append((chunk_id, similarity))
            
            refined_results.sort(key=lambda x: x[1], reverse=True)
            return refined_results[:k]
    
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
            self.compressed_data.clear()
            self.embeddings_cache.clear()
            self.inverted_lists = [[] for _ in range(self.num_clusters)]
    
    def get_statistics(self) -> Dict:
        with self._lock:
            # Calculate cluster distribution
            cluster_sizes = [len(inv_list) for inv_list in self.inverted_lists]
            non_empty_clusters = sum(1 for size in cluster_sizes if size > 0)
            avg_cluster_size = sum(cluster_sizes) / non_empty_clusters if non_empty_clusters > 0 else 0
            max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
            
            # Calculate compression
            original_size = self.dimension * 4 if self.dimension else 0
            compressed_size = self.num_subspaces * 1  # 1 byte per code
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
            
            return {
                "num_chunks": len(self.chunk_ids),
                "dimension": self.dimension,
                "num_coarse_clusters": self.num_clusters,
                "num_pq_subspaces": self.num_subspaces,
                "num_pq_clusters": self.num_pq_clusters,
                "nprobe": self.nprobe,
                "trained": self._trained,
                "non_empty_clusters": non_empty_clusters,
                "avg_cluster_size": f"{avg_cluster_size:.1f}",
                "max_cluster_size": max_cluster_size,
                "compression_ratio": f"{compression_ratio:.1f}x",
                "memory_per_vector": f"{self.num_subspaces} bytes (PQ) + cluster overhead"
            }
    
    def set_nprobe(self, nprobe: int) -> None:

        with self._lock:
            self.nprobe = min(nprobe, self.num_clusters)
    
    def get_cluster_distribution(self) -> Dict[str, int]:

        with self._lock:
            cluster_sizes = [len(inv_list) for inv_list in self.inverted_lists]
            
            return {
                "min_size": min(cluster_sizes) if cluster_sizes else 0,
                "max_size": max(cluster_sizes) if cluster_sizes else 0,
                "median_size": sorted(cluster_sizes)[len(cluster_sizes)//2] if cluster_sizes else 0,
                "empty_clusters": sum(1 for size in cluster_sizes if size == 0)
            }