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
- `brute_force` (default) - Exact search
- `kd_tree` - K-dimensional tree
- `lsh` - Locality-Sensitive Hashing
- `ivfpq` - Inverted File with Product Quantization

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

```bash
# Run all tests
pytest test/ -v

# Run specific test file
pytest test/test_api.py -v
pytest test/test_basic.py -v
```

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