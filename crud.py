from typing import List, Dict, Optional, Any
from datetime import datetime
from uuid import uuid4
import threading


class Chunk:
    def __init__(self, text: str, embedding: List[float], document_id: str, metadata: Optional[Dict[str, Any]] = None, chunk_id: Optional[str] = None):
        self.id = chunk_id or str(uuid4())
        self.text = text
        self.embedding = embedding
        self.document_id = document_id  
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self._lock = threading.RLock()

    def update_text(self, new_text: str):
        with self._lock:
            self.text = new_text
            self.updated_at = datetime.utcnow()

    def update_embedding(self, new_embedding: List[float]):
        with self._lock:
            self.embedding = new_embedding
            self.updated_at = datetime.utcnow()
    
    def update_metadata(self, new_metadata: Dict[str, Any]):
        with self._lock:
            self.metadata.update(new_metadata)
            self.updated_at = datetime.utcnow()
    
    def get_text(self) -> str:
        with self._lock:
            return self.text
        
    def get_embedding(self) -> List[float]:
        with self._lock:
            return self.embedding.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        with self._lock:
            return self.metadata.copy()
        
    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "text": self.text,
                "embedding": self.embedding,
                "document_id": self.document_id,
                "metadata": self.metadata,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat()
            }
        
class Document:
    def __init__(self, name: str, library_id: str, metadata: Optional[Dict[str, Any]] = None, document_id: Optional[str] = None):
        self.id = document_id or str(uuid4())
        self.name = name
        self.library_id = library_id
        self.metadata = metadata or {}
        self.chunks: List[str] = []
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self._lock = threading.RLock()

    def add_chunk(self, chunk_id: str):
        with self._lock:
            if chunk_id not in self.chunks:
                self.chunks.append(chunk_id)
                self.updated_at = datetime.utcnow()

    def remove_chunk(self, chunk_id: str):
        with self._lock:
            if chunk_id in self.chunks:
                self.chunks.remove(chunk_id)
                self.updated_at = datetime.utcnow()
                return True
            else:
                return False

    def get_chunks(self, chunk_id: str):
        with self._lock:
            return self.chunks.copy()
        
    def update_name(self, new_name: str):
        with self._lock:
            self.name = new_name
            self.updated_at = datetime.utcnow()
    
    def update_metadata(self, new_metadata: Dict[str, Any]):
        with self._lock:
            self.metadata =  new_metadata
            self.updated_at = datetime.utcnow()
    
    def get_metadata(self):
        with self._lock:
            return self.metadata.copy()
        
    def to_dict(self):
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "library_id": self.library_id,
                "metadata": self.metadata,
                "chunks": self.chunks,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat()
            }
        
class Library:
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None, library_id: Optional[str] = None):
        self.id = library_id or str(uuid4())
        self.name = name
        self.metadata = metadata or {}
        self.documents: List[str] = []
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self._lock = threading.RLock()

    def add_document(self, document_id: str):
        with self._lock:
            if document_id not in self.documents:
                self.documents.append(document_id)
                self.updated_at = datetime.utcnow()

    def remove_document(self, document_id: str):
        with self._lock:
            if document_id in self.documents:
                self.documents.remove(document_id)
                self.updated_at = datetime.utcnow()
                return True
            else:
                return False
    
    def get_documents(self):
        with self._lock:
            return self.documents.copy()
        
    def update_name(self, new_name: str):
        with self._lock:
            self.name = new_name
            self.updated_at = datetime.utcnow()

    def update_metadata(self, new_metadata: Dict[str, Any]):
        with self._lock:
            self.metadata = new_metadata
            self.updated_at = datetime.utcnow()
    
    def get_metadata(self):
        with self._lock:
            return self.metadata.copy()
        
    def to_dict(self):
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "metadata": self.metadata,
                "documents": self.documents,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat()
            }