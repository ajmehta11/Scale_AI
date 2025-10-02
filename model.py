from typing import List, Dict, Optional, Any
from datetime import datetime
from uuid import uuid4
import threading
from embeddings import Embeddings
from dotenv import load_dotenv
import os



def _utcnow() -> datetime:
    return datetime.utcnow()

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
embedding_model = Embeddings(api_key=api_key)

class Chunk:
    def __init__(self, text: str, embedding_model, document_id: str, metadata: Optional[Dict[str, Any]] = None, chunk_id: Optional[str] = None):
        self.id: str = chunk_id or str(uuid4())
        self.text: str = text
        self.document_id: str = document_id
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.created_at: datetime = _utcnow()
        self.updated_at: datetime = _utcnow()
        self._lock = threading.RLock()
        self._embedding_model = embedding_model
        self.embedding: List[float] = self._embedding_model.get_embedding(text)

    def update_text(self, new_text: str):
        with self._lock:
            self.text = new_text
            self.embedding = self._embedding_model.get_embedding(new_text)
            self.updated_at = _utcnow()

    # def update_embedding(self, new_embedding: List[float]):
    #     with self._lock:
    #         self.embedding = list(new_embedding)
    #         self.updated_at = _utcnow()

    def update_metadata(self, new_metadata: Dict[str, Any]):
        with self._lock:
            self.metadata.update(new_metadata or {})
            self.updated_at = _utcnow()

    def get_text(self):
        with self._lock:
            return self.text

    def get_embedding(self):
        with self._lock:
            return list(self.embedding)

    def get_metadata(self):
        with self._lock:
            return dict(self.metadata)

    def to_dict(self):
        with self._lock:
            return {
                "id": self.id,
                "text": self.text,
                "embedding": list(self.embedding),
                "document_id": self.document_id,
                "metadata": dict(self.metadata),
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
            }


class Document:
    def __init__(self, name: str, library_id: str, metadata: Optional[Dict[str, Any]] = None, document_id: Optional[str] = None):
        self.id: str = document_id or str(uuid4())
        self.name: str = name
        self.library_id: str = library_id
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.chunk_ids: List[str] = []
        self.created_at: datetime = _utcnow()
        self.updated_at: datetime = _utcnow()
        self._lock = threading.RLock()

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

    def get_metadata(self):
        with self._lock:
            return dict(self.metadata)

    def to_dict(self):
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


class Library:
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None, library_id: Optional[str] = None):
        self.id: str = library_id or str(uuid4())
        self.name: str = name
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.document_ids: List[str] = []
        self.created_at: datetime = _utcnow()
        self.updated_at: datetime = _utcnow()
        self._lock = threading.RLock()

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

    def get_metadata(self):
        with self._lock:
            return dict(self.metadata)

    def to_dict(self):
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "metadata": dict(self.metadata),
                "documents": list(self.document_ids),
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
            }