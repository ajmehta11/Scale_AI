import pytest
from fastapi.testclient import TestClient
import concurrent.futures


@pytest.fixture
def client(embedding_client):
    """Create test client"""
    import main
    from indexes.brute_force import BruteForceIndex
    from core.repositories import Repository
    
    index = BruteForceIndex("test")
    main.repo = Repository(index)
    
    return TestClient(main.app)


class TestAPI:
    """Test FastAPI endpoints"""
    
    def test_health_check(self, client):
        """Test health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_create_library(self, client):
        """Test creating library"""
        response = client.post(
            "/libraries",
            json={"name": "Test Library", "metadata": {"type": "test"}}
        )
        assert response.status_code == 201
        assert response.json()["name"] == "Test Library"
    
    def test_create_document(self, client):
        """Test creating document"""
        # Create library first
        lib_resp = client.post("/libraries", json={"name": "Lib"})
        lib_id = lib_resp.json()["id"]
        
        # Create document
        response = client.post(
            "/documents",
            json={"name": "Doc", "library_id": lib_id}
        )
        assert response.status_code == 201
        assert response.json()["name"] == "Doc"
    
    def test_create_chunk(self, client):
        """Test creating chunk"""
        # Setup
        lib_resp = client.post("/libraries", json={"name": "Lib"})
        lib_id = lib_resp.json()["id"]
        
        doc_resp = client.post(
            "/documents",
            json={"name": "Doc", "library_id": lib_id}
        )
        doc_id = doc_resp.json()["id"]
        
        # Create chunk
        response = client.post(
            "/chunks",
            json={"text": "Test text", "document_id": doc_id}
        )
        assert response.status_code == 201
        assert response.json()["text"] == "Test text"
        assert response.json()["has_embedding"] is True
    
    def test_search(self, client):
        """Test search endpoint"""
        # Setup
        lib_resp = client.post("/libraries", json={"name": "Lib"})
        lib_id = lib_resp.json()["id"]
        
        doc_resp = client.post(
            "/documents",
            json={"name": "Doc", "library_id": lib_id}
        )
        doc_id = doc_resp.json()["id"]
        
        # Add chunks
        client.post("/chunks", json={"text": "AI and ML", "document_id": doc_id})
        client.post("/chunks", json={"text": "databases", "document_id": doc_id})
        
        # Search
        response = client.post(
            "/search",
            json={"query": "artificial intelligence", "k": 2}
        )
        assert response.status_code == 200
        assert "results" in response.json()
    
    def test_stats(self, client):
        """Test stats endpoint"""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "num_libraries" in data
        assert "num_documents" in data
        assert "num_chunks" in data


class TestConcurrency:
    """Test concurrent operations"""

    def test_concurrent_chunk_creation(self, client, setup_library):
        """Test concurrent chunk creation"""
        lib_id, doc_id = setup_library

        def create_chunk(i):
            return client.post("/chunks", json={
                "text": f"chunk {i}",
                "document_id": doc_id
            })

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_chunk, i) for i in range(20)]
            results = [f.result() for f in futures]

        assert all(r.status_code == 201 for r in results)

        # Verify all chunks created
        chunks_resp = client.get(f"/chunks?document_id={doc_id}")
        assert len(chunks_resp.json()) == 20

    def test_concurrent_searches(self, client, setup_library):
        """Test concurrent search operations"""
        lib_id, doc_id = setup_library

        # Add some chunks
        for i in range(10):
            client.post("/chunks", json={
                "text": f"test chunk {i}",
                "document_id": doc_id
            })

        def search(query):
            return client.post("/search", json={"query": query, "k": 5})

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(search, f"query {i}") for i in range(20)]
            results = [f.result() for f in futures]

        assert all(r.status_code == 200 for r in results)

    def test_concurrent_updates(self, client, setup_library):
        """Test concurrent update operations"""
        lib_id, doc_id = setup_library
        chunk_resp = client.post("/chunks", json={
            "text": "original",
            "document_id": doc_id
        })
        chunk_id = chunk_resp.json()["id"]

        def update_chunk(i):
            return client.put(f"/chunks/{chunk_id}", json={
                "text": f"updated {i}",
                "recompute_embedding": True
            })

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(update_chunk, i) for i in range(10)]
            results = [f.result() for f in futures]

        # All should succeed (last writer wins)
        assert all(r.status_code == 200 for r in results)
