# Vector Database API

A FastAPI-based vector database with semantic search using Cohere embeddings. Supports multiple indexing strategies.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
```bash
export COHERE_API_KEY=your-api-key-here
```

### 3. Run Locally
```bash
python3 main.py
```
API runs at: http://localhost:8000

### 4. Run with Docker
```bash
docker build -t vector-db .
docker run -p 8000:8000 -e COHERE_API_KEY=your-key vector-db
```

## Index Types

Set via `INDEX_TYPE` environment variable:

### Brute Force (default)
- **Algorithm**: Exhaustive linear search with cosine similarity
- **Accuracy**: 100% - exact nearest neighbors
- **Insert**: O(1) - add to list
- **Delete**: O(n) - remove from list
- **Search**: O(n·d) - compare with all vectors
- **Space**: O(n·d) - store all embeddings
- **Best for**: Small datasets (< 10K vectors), when accuracy is critical

### KD-Tree
- **Algorithm**: K-dimensional tree with spatial partitioning
- **Accuracy**: ~95-99% - approximate search with pruning
- **Insert**: O(log n) amortized - marks tree for rebuild
- **Delete**: O(log n) amortized - marks tree for rebuild
- **Search**: O(log n) average, O(n) worst case - tree traversal with pruning
- **Space**: O(n·d) - tree nodes + embeddings
- **Best for**: Medium datasets (10K-100K vectors), low-to-medium dimensions

### LSH (Locality-Sensitive Hashing)
- **Algorithm**: Random hyperplane hashing with multiple tables
- **Accuracy**: ~85-95% - probabilistic nearest neighbors
- **Insert**: O(T·H) - hash into T tables with H-bit hashes
- **Delete**: O(T·H) - remove from hash buckets
- **Search**: O(T·B + C·d) - probe T buckets, compute C candidates
- **Space**: O(n·d + T·H·d) - embeddings + hyperplanes + hash tables
- **Parameters**: `num_tables=10`, `hash_size=12`
- **Best for**: Large datasets (100K-1M+ vectors), high dimensions, fast queries

### IVFPQ (Inverted File with Product Quantization)
- **Algorithm**: Coarse quantization + product quantization compression
- **Accuracy**: ~80-90% - compressed approximate search
- **Insert**: O(K·d + M·K') - find cluster + encode subvectors
- **Delete**: O(1) amortized - inverted list removal
- **Search**: O(K·d + P·L·M + R·d) - find nearest clusters + ADC scoring + refinement
  where ADC uses precomputed lookup tables for fast approximate distance
- **Space**: O(K·d + M·K'·(d/M) + n·M·log₂(K')) 
  ≈ O(K·d + M·K'·d/M + n·M) in practice
  = coarse centroids + PQ codebooks + compressed codes (each code is log₂(K') bits)
- **Parameters**: `num_clusters=100`, `num_subspaces=8`, `nprobe=10`
- **Best for**: Very large datasets (1M+ vectors), limited memory, acceptable accuracy tradeoff

**Notation**: n = number of vectors, d = embedding dimension, k = top-k results, T = hash tables, H = hash bits, K = coarse clusters, M = subspaces, K' = PQ clusters per subspace, P = clusters probed, L = avg list length, R = refinement candidates, B = avg bucket size, C = total candidates

```bash
docker run -p 8000:8000 -e COHERE_API_KEY=your-key -e INDEX_TYPE=lsh vector-db
```

## API Documentation

Interactive docs: http://localhost:8000/docs

### Create Library
```bash
curl -X POST http://localhost:8000/libraries \
  -H "Content-Type: application/json" \
  -d '{"name": "My Library"}'
```

### Create Document
```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"name": "My Doc", "library_id": "lib-id"}'
```

### Create Chunk
```bash
curl -X POST http://localhost:8000/chunks \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here", "document_id": "doc-id"}'
```

### Search
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your question", "k": 5}'
```

**With metadata filter:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your question",
    "k": 5,
    "metadata_filter": {"category": "tech"}
  }'
```

### Get Stats
```bash
curl http://localhost:8000/stats
```

## Testing

### Unit Tests
```bash
# Run all tests
pytest test/ -v

# Run specific test file
pytest test/test_api.py -v
pytest test/test_basic.py -v
pytest test/test_edge_cases.py -v
```

### Comprehensive Index Testing
Test all 4 index types (brute_force, kd_tree, lsh, ivfpq) with 300 chunks across multiple libraries and documents. The script automatically creates Docker containers for each index, loads data, runs searches, and compares performance.

```bash
export COHERE_API_KEY=your-api-key-here
bash test_all_indexes.sh
```

This will test each index with 3 libraries, 5 documents per library, and 20 chunks per document. Results include load times, search latency, and throughput comparison.

## Project Structure

```
.
├── main.py                 # FastAPI application
├── core/
│   ├── model.py           # Data models
│   └── repositories.py    # Repository layer
├── indexes/
│   ├── brute_force.py     # Exact search
│   ├── kd_tree.py         # KD-Tree index
│   ├── lsh.py             # LSH index
│   └── ivfpq.py           # IVFPQ index
├── services/
│   └── embeddings.py      # Cohere integration
├── test/
│   ├── conftest.py        # Test fixtures
│   ├── test_api.py        # API tests
│   └── test_basic.py      # Unit tests
├── Dockerfile             # Container config
└── docker-compose.yml     # Multi-index setup
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Database statistics |
| `/libraries` | GET/POST | List/create libraries |
| `/libraries/{id}` | GET/PUT/DELETE | Get/update/delete library |
| `/documents` | GET/POST | List/create documents |
| `/documents/{id}` | GET/PUT/DELETE | Get/update/delete document |
| `/chunks` | GET/POST | List/create chunks |
| `/chunks/{id}` | GET/PUT/DELETE | Get/update/delete chunk |
| `/search` | POST | Semantic search |
| `/snapshot/save` | POST | Save state to disk |
| `/snapshot/load` | POST | Load state from disk |

## Environment Variables

```bash
COHERE_API_KEY=your-key-here    # Required
INDEX_TYPE=brute_force          # Optional (default: brute_force)
```

## Design Decisions

### Architecture: Repository Pattern

```
API (main.py) → Repository (repositories.py) → Index (brute_force/kd_tree/lsh/ivfpq)
```

**Why?**
- Clean separation: API doesn't know about indexes
- Easy testing: Mock Repository or Index independently
- Swappable indexes: Change via `INDEX_TYPE` env var

### Thread Safety: RLock

**Used everywhere:** Index classes, Pydantic models, Repository

```python
# In indexes
with self._lock:
    self.chunk_ids.append(chunk_id)

# In Pydantic models  
with self._lock:
    object.__setattr__(self, 'text', new_text)

# In Repository
with self._lock:
    self.chunks[chunk_id] = chunk
```

**Why RLock (not Lock)?**
- **Reentrant**: Same thread can acquire multiple times
- **Prevents deadlocks**: `delete_document(cascade=True)` calls `delete_chunk()` - both need locks
- **Simple**: One lock type, easy to reason about

**Tradeoffs:**
- ✅ Correct, deadlock-free
- ✅ Good for multi-threaded workloads
- ❌ Coarse-grained (writes block everything)
- ❌ Single-process only

### Persistence: JSON Snapshots

**How it works:**
```python
snapshot = {
    "libraries": [...],
    "documents": [...], 
    "chunks": [...]  # with embeddings
}
```

**Saves on:**
1. Application shutdown
2. After mutations (library/doc/chunk changes)
3. Manual: `POST /snapshot/save`

**Tradeoffs:**
- ✅ Human-readable, easy to debug
- ✅ Simple backup (copy JSON file)
- ❌ Full state in memory (not for 100M+ vectors)
- ❌ No incremental saves

### Index Selection

**Decision tree:**
- < 10K vectors: `brute_force` (exact, simple)
- < 100K vectors: `kd_tree` (exact, faster)  
- < 1M vectors: `lsh` (approximate, fast)
- \> 1M vectors: `ivfpq` (compressed, scalable)

**Implementation:** Factory pattern in `main.py` reads `INDEX_TYPE` env var

### Why No FAISS/Pinecone/ChromaDB?

**Requirement:** Implement algorithms from scratch

**Used:**
- ✅ `numpy` - math operations (allowed)
- ✅ `threading` - standard library

**Implemented from scratch:**
- Cosine similarity
- K-means clustering (IVFPQ)
- Random projection (LSH)
- KD-tree construction
- Product quantization