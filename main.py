from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import os
import json
from pathlib import Path

import core.model as model
from core.repositories import Repository
from indexes.brute_force import BruteForceIndex
from indexes.kd_tree import KDTreeIndex
from indexes.lsh import LSHIndex
from indexes.ivfpq import IVFPQIndex
from services.embeddings import Embeddings



class LibraryCreate(BaseModel):
    name: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class LibraryUpdate(BaseModel):
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LibraryResponse(BaseModel):
    id: str
    name: str
    metadata: Dict[str, Any]
    documents: List[str]
    created_at: str
    updated_at: str


class DocumentCreate(BaseModel):
    name: str
    library_id: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DocumentUpdate(BaseModel):
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    id: str
    name: str
    library_id: str
    metadata: Dict[str, Any]
    chunks: List[str]
    created_at: str
    updated_at: str


class ChunkCreate(BaseModel):
    text: str
    document_id: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChunkUpdate(BaseModel):
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    recompute_embedding: bool = True


class ChunkResponse(BaseModel):
    id: str
    text: str
    document_id: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    has_embedding: bool


class SearchRequest(BaseModel):
    query: str
    library_id: Optional[str] = None
    k: int = Field(default=10, ge=1, le=100)
    metadata_filter: Optional[Dict[str, Any]] = None


class SearchResult(BaseModel):
    chunk_id: str
    score: float
    text: str
    document_id: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResult]
    count: int


class StatsResponse(BaseModel):
    num_libraries: int
    num_documents: int
    num_chunks: int


repo: Optional[Repository] = None
persistence_path = Path("./data")


def get_index(index_type: str = "brute_force"):
    """Factory function to create the appropriate index"""
    if index_type == "brute_force":
        return BruteForceIndex("main-library")
    elif index_type == "kd_tree":
        return KDTreeIndex("main-library")
    elif index_type == "lsh":
        return LSHIndex("main-library", num_tables=10, hash_size=12)
    elif index_type == "ivfpq":
        return IVFPQIndex("main-library", num_clusters=100, num_subspaces=8)
    else:
        raise ValueError(f"Unknown index type: {index_type}")


def save_to_disk():
    """Save repository state to disk"""
    if repo is None:
        return
    
    persistence_path.mkdir(exist_ok=True)
    snapshot = repo.to_snapshot()
    
    with open(persistence_path / "snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)


def load_from_disk():
    """Load repository state from disk"""
    snapshot_file = persistence_path / "snapshot.json"
    
    if snapshot_file.exists():
        with open(snapshot_file) as f:
            snapshot = json.load(f)
        repo.load_from_snapshot(snapshot)
        print(f"Loaded snapshot: {repo.get_stats()}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global repo
    
    # Startup
    print("Starting Vector Database API...")
    
    # Initialize embedding client
    api_key = os.getenv("COHERE_API_KEY")
    embedding_client = Embeddings(api_key)
    model.init_embedding_client(embedding_client)
    print("Embedding client initialized")
    
    # Initialize repository with selected index
    index_type = os.getenv("INDEX_TYPE", "brute_force")
    index = get_index(index_type)
    repo = Repository(index)
    print(f"Repository initialized with {index_type} index")
    
    # Load from disk if available
    load_from_disk()
    
    yield
    
    # Shutdown
    print("Shutting down Vector Database API...")
    save_to_disk()
    print("State saved to disk")


app = FastAPI(
    title="Vector Database API",
    description="A REST API for indexing and querying documents in a vector database",
    version="1.0.0",
    lifespan=lifespan
)


# Health Check
@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "message": "Vector DB API is running"}


@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
def get_stats():
    return repo.get_stats()


# Library Endpoints
@app.post("/libraries", response_model=LibraryResponse, status_code=status.HTTP_201_CREATED, tags=["Libraries"])
def create_library(library: LibraryCreate):
    try:
        lib = model.Library(name=library.name, metadata=library.metadata)
        repo.add_library(lib)
        save_to_disk()
        return lib.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/libraries", response_model=List[LibraryResponse], tags=["Libraries"])
def list_libraries():
    libraries = repo.list_libraries()
    return [lib.to_dict() for lib in libraries]


@app.get("/libraries/{library_id}", response_model=LibraryResponse, tags=["Libraries"])
def get_library(library_id: str):
    lib = repo.get_library(library_id)
    if lib is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Library {library_id} not found")
    return lib.to_dict()


@app.put("/libraries/{library_id}", response_model=LibraryResponse, tags=["Libraries"])
def update_library(library_id: str, update: LibraryUpdate):
    try:
        if update.name:
            repo.update_library_name(library_id, update.name)
        if update.metadata:
            repo.update_library_metadata(library_id, update.metadata)
        
        lib = repo.get_library(library_id)
        save_to_disk()
        return lib.to_dict()
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Library {library_id} not found")


@app.delete("/libraries/{library_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Libraries"])
def delete_library(library_id: str, cascade: bool = True):
    result = repo.delete_library(library_id, cascade=cascade)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Library {library_id} not found")
    save_to_disk()


# Document Endpoints
@app.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED, tags=["Documents"])
def create_document(document: DocumentCreate):
    try:
        doc = model.Document(name=document.name, library_id=document.library_id, metadata=document.metadata)
        repo.add_document(doc)
        save_to_disk()
        return doc.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/documents", response_model=List[DocumentResponse], tags=["Documents"])
def list_documents(library_id: Optional[str] = None):
    documents = repo.list_documents(library_id=library_id)
    return [doc.to_dict() for doc in documents]


@app.get("/documents/{document_id}", response_model=DocumentResponse, tags=["Documents"])
def get_document(document_id: str):
    doc = repo.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {document_id} not found")
    return doc.to_dict()


@app.put("/documents/{document_id}", response_model=DocumentResponse, tags=["Documents"])
def update_document(document_id: str, update: DocumentUpdate):
    try:
        if update.name:
            repo.update_document_name(document_id, update.name)
        if update.metadata:
            repo.update_document_metadata(document_id, update.metadata)
        
        doc = repo.get_document(document_id)
        save_to_disk()
        return doc.to_dict()
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {document_id} not found")


@app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Documents"])
def delete_document(document_id: str, cascade: bool = True):
    result = repo.delete_document(document_id, cascade=cascade)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document {document_id} not found")
    save_to_disk()


@app.post("/chunks", response_model=ChunkResponse, status_code=status.HTTP_201_CREATED, tags=["Chunks"])
def create_chunk(chunk: ChunkCreate):
    try:
        created_chunk = repo.add_chunk_from_text(
            chunk.text,
            chunk.document_id,
            metadata=chunk.metadata
        )
        save_to_disk()
        
        response = created_chunk.to_dict()
        response["has_embedding"] = created_chunk.get_embedding() is not None
        return response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/chunks", response_model=List[ChunkResponse], tags=["Chunks"])
def list_chunks(document_id: Optional[str] = None, library_id: Optional[str] = None):
    chunks = repo.list_chunks(document_id=document_id, library_id=library_id)
    results = []
    for c in chunks:
        data = c.to_dict()
        data["has_embedding"] = c.get_embedding() is not None
        results.append(data)
    return results


@app.get("/chunks/{chunk_id}", response_model=ChunkResponse, tags=["Chunks"])
def get_chunk(chunk_id: str):
    chunk = repo.get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chunk {chunk_id} not found")
    
    response = chunk.to_dict()
    response["has_embedding"] = chunk.get_embedding() is not None
    return response


@app.put("/chunks/{chunk_id}", response_model=ChunkResponse, tags=["Chunks"])
def update_chunk(chunk_id: str, update: ChunkUpdate):
    try:
        if update.text:
            chunk = repo.update_chunk_text(
                chunk_id,
                update.text,
                recompute_embedding=update.recompute_embedding
            )
        
        if update.metadata:
            chunk = repo.get_chunk(chunk_id)
            if chunk is None:
                raise KeyError(chunk_id)
            chunk.update_metadata(update.metadata)
        
        chunk = repo.get_chunk(chunk_id)
        save_to_disk()
        
        response = chunk.to_dict()
        response["has_embedding"] = chunk.get_embedding() is not None
        return response
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chunk {chunk_id} not found")


@app.delete("/chunks/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Chunks"])
def delete_chunk(chunk_id: str):
    result = repo.delete_chunk(chunk_id)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Chunk {chunk_id} not found")
    save_to_disk()


@app.post("/search", response_model=SearchResponse, tags=["Search"])
def search(search_request: SearchRequest):
    """Search for similar chunks using vector similarity"""
    try:
        results = repo.search(
            query_text=search_request.query,
            library_id=search_request.library_id,
            k=search_request.k,
            metadata_filter=search_request.metadata_filter
        )
        
        search_results = []
        for chunk_id, score in results:
            chunk = repo.get_chunk(chunk_id)
            if chunk:
                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        score=score,
                        text=chunk.text,
                        document_id=chunk.document_id,
                        metadata=chunk.metadata
                    )
                )
        
        return SearchResponse(
            results=search_results,
            count=len(search_results)
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/snapshot/save", tags=["Persistence"])
def save_snapshot():
    save_to_disk()
    return {"message": "Snapshot saved successfully", "stats": repo.get_stats()}


@app.post("/snapshot/load", tags=["Persistence"])
def load_snapshot():
    load_from_disk()
    return {"message": "Snapshot loaded successfully", "stats": repo.get_stats()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)