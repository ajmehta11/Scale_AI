import pytest
import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import core.model as model
from core.repositories import Repository
from indexes.brute_force import BruteForceIndex


class MockEmbeddingClient:
    
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
    
    def get_embedding(self, text: str) -> List[float]:
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        
        embedding = []
        for i in range(self.dimension):
            val = ((hash_val + i) % 1000) / 1000.0
            embedding.append(val)
        
        # Normalize
        norm = sum(x**2 for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding


@pytest.fixture(autouse=True)
def reset_embedding_client():
    model._CLIENT_INITIALIZED = False
    model._EMBEDDING_CLIENT = None
    yield
    model._CLIENT_INITIALIZED = False
    model._EMBEDDING_CLIENT = None


@pytest.fixture
def embedding_client():
    """Provide mock embedding client"""
    client = MockEmbeddingClient(dimension=128)
    model.init_embedding_client(client)
    return client


@pytest.fixture(params=["brute_force", "kd_tree", "lsh", "ivfpq"])
def repository(request, embedding_client):
    """Provide repository with different index types"""
    if request.param == "brute_force":
        from indexes.brute_force import BruteForceIndex
        index = BruteForceIndex("test-lib")
    elif request.param == "kd_tree":
        from indexes.kd_tree import KDTreeIndex
        index = KDTreeIndex("test-lib")
    elif request.param == "lsh":
        from indexes.lsh import LSHIndex
        index = LSHIndex("test-lib", num_tables=5, hash_size=8, seed=42)
    elif request.param == "ivfpq":
        from indexes.ivfpq import IVFPQIndex
        index = IVFPQIndex("test-lib", num_clusters=10, num_subspaces=4, seed=42)

    return Repository(index)


@pytest.fixture
def client(embedding_client):
    """Create test client for API tests"""
    from fastapi.testclient import TestClient
    import main
    from indexes.brute_force import BruteForceIndex
    from core.repositories import Repository

    index = BruteForceIndex("test")
    main.repo = Repository(index)

    return TestClient(main.app)


@pytest.fixture
def setup_library(client):
    """Helper fixture to create library and document"""
    lib_resp = client.post("/libraries", json={"name": "Test Lib"})
    lib_id = lib_resp.json()["id"]

    doc_resp = client.post("/documents", json={
        "name": "Test Doc",
        "library_id": lib_id
    })
    doc_id = doc_resp.json()["id"]

    return lib_id, doc_id