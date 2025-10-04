import pytest
import core.model as model
from core.repositories import Repository  


class TestModels:
    
    def test_create_library(self):
        lib = model.Library(name="Test Library", metadata={"type": "test"})
        assert lib.name == "Test Library"
        assert lib.id is not None
    
    def test_create_document(self):
        doc = model.Document(
            name="Test Doc",
            library_id="lib-123",
            metadata={"author": "Alice"}
        )
        assert doc.name == "Test Doc"
        assert doc.library_id == "lib-123"
    
    def test_create_chunk(self, embedding_client):
        """Test chunk creation"""
        chunk = model.Chunk.from_text(
            text="This is a test",
            document_id="doc-123"
        )
        assert chunk.text == "This is a test"
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 128
    
    def test_chunk_update_text(self, embedding_client):
        """Test updating chunk text"""
        chunk = model.Chunk.from_text("original", document_id="doc-1")
        original_emb = chunk.get_embedding()
        
        chunk.update_text("updated", recompute=True)
        
        assert chunk.text == "updated"
        assert chunk.get_embedding() != original_emb
    
    def test_serialization(self, embedding_client):
        """Test to_dict and from_dict"""
        chunk = model.Chunk.from_text("test", document_id="doc-1")
        chunk_dict = chunk.to_dict()
        
        new_chunk = model.Chunk.from_dict(chunk_dict)
        assert new_chunk.id == chunk.id
        assert new_chunk.text == chunk.text


class TestRepository:
    """Test repository operations"""
    
    def test_add_library(self, repository):
        """Test adding library"""
        lib = model.Library(name="Test Lib")
        repository.add_library(lib)
        
        retrieved = repository.get_library(lib.id)
        assert retrieved is not None
        assert retrieved.name == "Test Lib"
    
    def test_add_document(self, repository):
        """Test adding document"""
        lib = model.Library(name="Test Lib")
        repository.add_library(lib)
        
        doc = model.Document(name="Test Doc", library_id=lib.id)
        repository.add_document(doc)
        
        retrieved = repository.get_document(doc.id)
        assert retrieved.name == "Test Doc"
    
    def test_add_chunk(self, repository, embedding_client):
        """Test adding chunk"""
        lib = model.Library(name="Lib")
        repository.add_library(lib)
        
        doc = model.Document(name="Doc", library_id=lib.id)
        repository.add_document(doc)
        
        chunk = repository.add_chunk_from_text("Sample text", doc.id)
        
        assert chunk.text == "Sample text"
        assert chunk.get_embedding() is not None
    
    def test_search(self, repository, embedding_client):
        """Test search functionality"""
        lib = model.Library(name="Search Lib")
        repository.add_library(lib)
        
        doc = model.Document(name="Doc", library_id=lib.id)
        repository.add_document(doc)
        
        # Add chunks
        repository.add_chunk_from_text("machine learning", doc.id)
        repository.add_chunk_from_text("deep learning", doc.id)
        repository.add_chunk_from_text("database systems", doc.id)
        
        # Search
        results = repository.search("artificial intelligence", k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    
    def test_delete_cascade(self, repository, embedding_client):
        """Test cascade delete"""
        lib = model.Library(name="Lib")
        repository.add_library(lib)
        
        doc = model.Document(name="Doc", library_id=lib.id)
        repository.add_document(doc)
        
        chunk1 = repository.add_chunk_from_text("text1", doc.id)
        chunk2 = repository.add_chunk_from_text("text2", doc.id)
        
        # Delete document with cascade
        repository.delete_document(doc.id, cascade=True)
        
        assert repository.get_document(doc.id) is None
        assert repository.get_chunk(chunk1.id) is None
        assert repository.get_chunk(chunk2.id) is None
    
    def test_snapshot(self, repository, embedding_client):
        """Test snapshot save and load"""
        lib = model.Library(name="Snapshot Lib")
        repository.add_library(lib)
        
        doc = model.Document(name="Doc", library_id=lib.id)
        repository.add_document(doc)
        
        chunk = repository.add_chunk_from_text("test", doc.id)
        
        # Save snapshot
        snapshot = repository.to_snapshot()
        
        # Load in new repository
        from indexes.brute_force import BruteForceIndex
        new_repo = Repository(BruteForceIndex("new"))
        new_repo.load_from_snapshot(snapshot)
        
        # Verify
        assert new_repo.get_stats()["num_libraries"] == 1
        assert new_repo.get_stats()["num_chunks"] == 1


class TestIndexes:
    """Test index operations"""
    
    def test_brute_force_search(self, embedding_client):
        """Test brute force index"""
        from indexes.brute_force import BruteForceIndex
        
        index = BruteForceIndex("test")
        
        emb1 = embedding_client.get_embedding("hello")
        emb2 = embedding_client.get_embedding("world")
        
        index.insert("chunk-1", emb1)
        index.insert("chunk-2", emb2)
        
        chunk1 = model.Chunk(text="hello", document_id="doc-1", embedding=emb1)
        chunk2 = model.Chunk(text="world", document_id="doc-1", embedding=emb2)
        chunks_map = {"chunk-1": chunk1, "chunk-2": chunk2}
        
        results = index.search(emb1, chunks_map, k=2)
        
        assert len(results) == 2
        assert results[0][0] == "chunk-1"  # Most similar
    
    def test_index_delete(self, embedding_client):
        """Test deleting from index"""
        from indexes.brute_force import BruteForceIndex
        
        index = BruteForceIndex("test")
        emb = embedding_client.get_embedding("test")
        
        index.insert("chunk-1", emb)
        assert index.get_size() == 1
        
        result = index.delete("chunk-1")
        assert result is True
        assert index.get_size() == 0
    
    def test_index_update(self, embedding_client):
        """Test updating index"""
        from indexes.brute_force import BruteForceIndex
        
        index = BruteForceIndex("test")
        emb1 = embedding_client.get_embedding("original")
        emb2 = embedding_client.get_embedding("updated")
        
        index.insert("chunk-1", emb1)
        result = index.update("chunk-1", emb2)
        
        assert result is True