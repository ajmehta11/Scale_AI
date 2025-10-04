from __future__ import annotations

from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from uuid import uuid4
import threading
import math

_EMBEDDING_CLIENT: Optional[Any] = None


def init_embedding_client(client: Any) -> None:
    global _EMBEDDING_CLIENT
    _EMBEDDING_CLIENT = client


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


class Chunk:

    def __init__(self, text: str, document_id: str, chunk_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, embedding: Optional[List[float]] = None):
        self.id: str = chunk_id or str(uuid4())
        self.text: str = text
        self.document_id: str = document_id
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.created_at: datetime = _utcnow()
        self.updated_at: datetime = _utcnow()
        self._lock = threading.RLock()
        self._embedding: Optional[List[float]] = None
        if embedding is not None:
            self._set_embedding_private(_validate_embedding(list(embedding)))

    @classmethod
    def from_text(cls, text: str, document_id: str) -> "Chunk":
        client = get_embedding_client()
        emb = client.get_embedding(text)
        c = cls(text=text, document_id=document_id)
        c._set_embedding_private(_validate_embedding(list(emb)))
        return c

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        c = cls(
            text=d.get("text", ""),
            document_id=d.get("document_id", ""),
            chunk_id=d.get("id"),
            metadata=d.get("metadata", {}),
            embedding=None,
        )
        emb = d.get("embedding")
        if emb is not None:
            c._set_embedding_private(_validate_embedding(list(emb)))
        try:
            if d.get("created_at"):
                c.created_at = datetime.fromisoformat(d["created_at"])
            if d.get("updated_at"):
                c.updated_at = datetime.fromisoformat(d["updated_at"])
        except Exception:
            pass
        return c

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "text": self.text,
                "embedding": list(self._embedding) if self._embedding is not None else None,
                "document_id": self.document_id,
                "metadata": dict(self.metadata),
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
            }

    def _set_embedding_private(self, emb: List[float]) -> None:
        with self._lock:
            self._embedding = list(emb)
            self.updated_at = _utcnow()

    def compute_embedding(self) -> List[float]:
 
        client = get_embedding_client()
        emb = client.get_embedding(self.text)
        emb = _validate_embedding(list(emb))
        self._set_embedding_private(emb)
        return list(emb)

    # --- public mutators ---
    def update_text(self, new_text: str, recompute: bool = True) -> None:

        with self._lock:
            self.text = new_text
            self.updated_at = _utcnow()
        if recompute:
            self.compute_embedding()

    def update_embedding(self, new_embedding: List[float]) -> None:

        emb = _validate_embedding(list(new_embedding))
        self._set_embedding_private(emb)

    def update_metadata(self, new_metadata: Dict[str, Any]) -> None:
        with self._lock:
            self.metadata.update(new_metadata or {})
            self.updated_at = _utcnow()

    def get_text(self) -> str:
        with self._lock:
            return self.text

    def get_embedding(self) -> Optional[List[float]]:
        with self._lock:
            return list(self._embedding) if self._embedding is not None else None

    def get_metadata(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.metadata)


class Document:
    def __init__( self, name: str, library_id: str, metadata: Optional[Dict[str, Any]] = None, document_id: Optional[str] = None,):
        self.id: str = document_id or str(uuid4())
        self.name: str = name
        self.library_id: str = library_id
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.chunk_ids: List[str] = []
        self.created_at: datetime = _utcnow()
        self.updated_at: datetime = _utcnow()
        self._lock = threading.RLock()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Document":
        doc = cls(name=d.get("name", ""), library_id=d.get("library_id", ""), document_id=d.get("id"))
        if d.get("metadata"):
            doc.metadata = dict(d.get("metadata") or {})
        for cid in d.get("chunks", []):
            doc.chunk_ids.append(cid)
        try:
            if d.get("created_at"):
                doc.created_at = datetime.fromisoformat(d["created_at"])
            if d.get("updated_at"):
                doc.updated_at = datetime.fromisoformat(d["updated_at"])
        except Exception:
            pass
        return doc

    def to_dict(self) -> Dict[str, Any]:
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

    def add_chunk(self, chunk_id: str):
        with self._lock:
            if chunk_id not in self.chunk_ids:
                self.chunk_ids.append(chunk_id)
                self.updated_at = _utcnow()

    def remove_chunk(self, chunk_id: str):
        with self._lock:
            if chunk_id in self.chunk_ids:
                self.chunk_ids.remove(chunk_id)
                self.updated_at = _utcnow()
                return True
            return False

    def get_chunks(self):
        with self._lock:
            return list(self.chunk_ids)

    def update_name(self, new_name: str):
        with self._lock:
            self.name = new_name
            self.updated_at = _utcnow()

    def update_metadata(self, new_metadata: Dict[str, Any]):
        with self._lock:
            self.metadata.update(new_metadata or {})
            self.updated_at = _utcnow()


class Library:
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None, library_id: Optional[str] = None):
        self.id: str = library_id or str(uuid4())
        self.name: str = name
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.document_ids: List[str] = []
        self.created_at: datetime = _utcnow()
        self.updated_at: datetime = _utcnow()
        self._lock = threading.RLock()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Library":
        lib = cls(name=d.get("name", ""), metadata=d.get("metadata", {}), library_id=d.get("id"))
        for did in d.get("documents", []):
            lib.document_ids.append(did)
        try:
            if d.get("created_at"):
                lib.created_at = datetime.fromisoformat(d["created_at"])
            if d.get("updated_at"):
                lib.updated_at = datetime.fromisoformat(d["updated_at"])
        except Exception:
            pass
        return lib

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

    def add_document(self, document_id: str):
        with self._lock:
            if document_id not in self.document_ids:
                self.document_ids.append(document_id)
                self.updated_at = _utcnow()

    def remove_document(self, document_id: str):
        with self._lock:
            if document_id in self.document_ids:
                self.document_ids.remove(document_id)
                self.updated_at = _utcnow()
                return True
            return False

    def get_documents(self):
        with self._lock:
            return list(self.document_ids)

    def update_name(self, new_name: str):
        with self._lock:
            self.name = new_name
            self.updated_at = _utcnow()

    def update_metadata(self, new_metadata: Dict[str, Any]):
        with self._lock:
            self.metadata.update(new_metadata or {})
            self.updated_at = _utcnow()
