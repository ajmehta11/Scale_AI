from __future__ import annotations

from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from uuid import uuid4
import threading
import math
from pydantic import BaseModel, Field, field_validator, PrivateAttr

_EMBEDDING_CLIENT: Optional[Any] = None
_CLIENT_INITIALIZED: bool = False


def init_embedding_client(client: Any) -> None:
    global _EMBEDDING_CLIENT, _CLIENT_INITIALIZED
    
    if _CLIENT_INITIALIZED:
        raise RuntimeError(
            "Embedding client already initialized. "
            "This should only be called once at startup."
        )
    
    _EMBEDDING_CLIENT = client
    _CLIENT_INITIALIZED = True


def get_embedding_client() -> Any:
    if _EMBEDDING_CLIENT is None:
        raise RuntimeError("Embedding client not initialized. Call init_embedding_client(...) first.")
    return _EMBEDDING_CLIENT


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _validate_embedding(emb: List[float]) -> List[float]:
    if not isinstance(emb, (list, tuple)):
        raise ValueError("Embedding must be a list of floats.")
    validated = []
    for e in emb:
        if not isinstance(e, (float, int)):
            raise ValueError("Embedding elements must be numeric.")
        if not math.isfinite(float(e)):
            raise ValueError("Embedding contains non-finite value.")
        validated.append(float(e))
    return validated


class Chunk(BaseModel):

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    document_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    embedding: Optional[List[float]] = None
    
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True
    }
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding_field(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return None
        return _validate_embedding(list(v))
    
    @classmethod
    def from_text(cls, text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> "Chunk":
        client = get_embedding_client()
        emb = client.get_embedding(text)
        validated_emb = _validate_embedding(list(emb))
        
        return cls(
            text=text,
            document_id=document_id,
            metadata=metadata or {},
            embedding=validated_emb
        )
    
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "text": self.text,
                "embedding": list(self.embedding) if self.embedding is not None else None,
                "document_id": self.document_id,
                "metadata": dict(self.metadata),
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
            }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        created_at = d.get("created_at")
        updated_at = d.get("updated_at")
        
        # Parse datetime strings if present
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        return cls(
            id=d.get("id", str(uuid4())),
            text=d.get("text", ""),
            document_id=d.get("document_id", ""),
            metadata=d.get("metadata", {}),
            embedding=d.get("embedding"),
            created_at=created_at or _utcnow(),
            updated_at=updated_at or _utcnow()
        )
    
    def _set_embedding_private(self, emb: List[float]) -> None:
        with self._lock:
            object.__setattr__(self, 'embedding', list(emb))
            object.__setattr__(self, 'updated_at', _utcnow())
    
    def compute_embedding(self) -> List[float]:
        client = get_embedding_client()
        emb = client.get_embedding(self.text)
        emb = _validate_embedding(list(emb))
        self._set_embedding_private(emb)
        return list(emb)
    
    def update_text(self, new_text: str, recompute: bool = True) -> None:
        with self._lock:
            object.__setattr__(self, 'text', new_text)
            object.__setattr__(self, 'updated_at', _utcnow())
        
        if recompute:
            self.compute_embedding()
    
    def update_embedding(self, new_embedding: List[float]) -> None:
        emb = _validate_embedding(list(new_embedding))
        self._set_embedding_private(emb)
    
    def update_metadata(self, new_metadata: Dict[str, Any]) -> None:
        with self._lock:
            self.metadata.update(new_metadata or {})
            object.__setattr__(self, 'updated_at', _utcnow())
    
    def get_text(self) -> str:
        with self._lock:
            return self.text
    
    def get_embedding(self) -> Optional[List[float]]:
        with self._lock:
            return list(self.embedding) if self.embedding is not None else None
    
    def get_metadata(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.metadata)


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    library_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True
    }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Document":
        """Create a document from a dictionary representation."""
        created_at = d.get("created_at")
        updated_at = d.get("updated_at")
        
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        return cls(
            id=d.get("id", str(uuid4())),
            name=d.get("name", ""),
            library_id=d.get("library_id", ""),
            metadata=d.get("metadata", {}),
            chunk_ids=list(d.get("chunks", [])),
            created_at=created_at or _utcnow(),
            updated_at=updated_at or _utcnow()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary representation."""
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "library_id": self.library_id,
                "metadata": dict(self.metadata),
                "chunks": list(self.chunk_ids),
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
            }
    
    def add_chunk(self, chunk_id: str) -> None:
        """Add a chunk ID to this document."""
        with self._lock:
            if chunk_id not in self.chunk_ids:
                self.chunk_ids.append(chunk_id)
                object.__setattr__(self, 'updated_at', _utcnow())
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk ID from this document."""
        with self._lock:
            if chunk_id in self.chunk_ids:
                self.chunk_ids.remove(chunk_id)
                object.__setattr__(self, 'updated_at', _utcnow())
                return True
            return False
    
    def get_chunks(self) -> List[str]:
        """Thread-safe getter for chunk IDs."""
        with self._lock:
            return list(self.chunk_ids)
    
    def update_name(self, new_name: str) -> None:
        """Update the document's name."""
        with self._lock:
            object.__setattr__(self, 'name', new_name)
            object.__setattr__(self, 'updated_at', _utcnow())
    
    def update_metadata(self, new_metadata: Dict[str, Any]) -> None:
        """Update the document's metadata."""
        with self._lock:
            self.metadata.update(new_metadata or {})
            object.__setattr__(self, 'updated_at', _utcnow())


class Library(BaseModel):

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)
    
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True
    }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Library":
        created_at = d.get("created_at")
        updated_at = d.get("updated_at")
        
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        return cls(
            id=d.get("id", str(uuid4())),
            name=d.get("name", ""),
            metadata=d.get("metadata", {}),
            document_ids=list(d.get("documents", [])),
            created_at=created_at or _utcnow(),
            updated_at=updated_at or _utcnow()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "metadata": dict(self.metadata),
                "documents": list(self.document_ids),
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
            }
    
    def add_document(self, document_id: str) -> None:
        with self._lock:
            if document_id not in self.document_ids:
                self.document_ids.append(document_id)
                object.__setattr__(self, 'updated_at', _utcnow())
    
    def remove_document(self, document_id: str) -> bool:
        with self._lock:
            if document_id in self.document_ids:
                self.document_ids.remove(document_id)
                object.__setattr__(self, 'updated_at', _utcnow())
                return True
            return False
    
    def get_documents(self) -> List[str]:
        with self._lock:
            return list(self.document_ids)
    
    def update_name(self, new_name: str) -> None:
        with self._lock:
            object.__setattr__(self, 'name', new_name)
            object.__setattr__(self, 'updated_at', _utcnow())
    
    def update_metadata(self, new_metadata: Dict[str, Any]) -> None:
        with self._lock:
            self.metadata.update(new_metadata or {})
            object.__setattr__(self, 'updated_at', _utcnow())