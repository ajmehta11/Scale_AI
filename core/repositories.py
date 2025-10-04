from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from threading import RLock

if TYPE_CHECKING:
    from model import Chunk, Document, Library

import model


class Repository:
    def __init__(self, index: Any):
        self._lock = RLock()
        self.index = index
        self.chunks: Dict[str, model.Chunk] = {}
        self.documents: Dict[str, model.Document] = {}
        self.libraries: Dict[str, model.Library] = {}


    def _index_insert(self, chunk_id: str, embedding: List[float]) -> None:
        self.index.insert(chunk_id, embedding)

    def _index_delete(self, chunk_id: str) -> bool:
        return bool(self.index.delete(chunk_id))

    def _index_update(self, chunk_id: str, embedding: List[float]) -> bool:
        return bool(self.index.update(chunk_id, embedding))

    def add_chunk_from_text(
        self,
        text: str,
        document_id: str,
        chunk_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> model.Chunk:
        chunk = model.Chunk(text, document_id, chunk_id, metadata)
        chunk.compute_embedding()
        self.add_chunk(chunk)
        return chunk

    def add_chunk(self, chunk: model.Chunk) -> None:
        emb = chunk.get_embedding()
        if emb is None:
            raise ValueError("Chunk has no embedding; compute it before adding.")

        with self._lock:
            if chunk.id in self.chunks:
                raise ValueError(f"Chunk {chunk.id} already exists")
            if chunk.document_id not in self.documents:
                raise ValueError(f"Document {chunk.document_id} not found")

            self.chunks[chunk.id] = chunk
            self.documents[chunk.document_id].add_chunk(chunk.id)

        self._index_insert(chunk.id, emb)

    def add_chunks_many(self, chunks: List[model.Chunk]) -> None:
        with self._lock:
            for c in chunks:
                if c.id in self.chunks:
                    raise ValueError(f"Chunk {c.id} already exists")
                if c.document_id not in self.documents:
                    raise ValueError(f"Document {c.document_id} not found")
                self.chunks[c.id] = c
                self.documents[c.document_id].add_chunk(c.id)

        for c in chunks:
            emb = c.get_embedding()
            if emb is None:
                raise ValueError(f"Chunk {c.id} has no embedding")
            self._index_insert(c.id, emb)

    def get_chunk(self, chunk_id: str) -> Optional[model.Chunk]:
        with self._lock:
            return self.chunks.get(chunk_id)

    def list_chunks(self, document_id: Optional[str] = None, library_id: Optional[str] = None) -> List[model.Chunk]:
        with self._lock:
            all_chunks = list(self.chunks.values())
            if document_id:
                all_chunks = [c for c in all_chunks if c.document_id == document_id]
            if library_id:
                docs_in_lib = {did for did, doc in self.documents.items() if doc.library_id == library_id}
                all_chunks = [c for c in all_chunks if c.document_id in docs_in_lib]
            return all_chunks

    def update_chunk_text(self, chunk_id: str, new_text: str, recompute_embedding: bool = True) -> model.Chunk:
        with self._lock:
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                raise KeyError(chunk_id)

        chunk.update_text(new_text, recompute=recompute_embedding)
        if recompute_embedding:
            emb = chunk.get_embedding()
            if emb is None:
                raise RuntimeError("Embedding missing after update")
            self._index_update(chunk_id, emb)

        return chunk

    def update_chunk_embedding(self, chunk_id: str, new_embedding: List[float]) -> None:
        with self._lock:
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                raise KeyError(chunk_id)
            chunk.update_embedding(new_embedding)

        emb = chunk.get_embedding()
        self._index_update(chunk_id, emb)

    def delete_chunk(self, chunk_id: str) -> bool:
        deleted = self._index_delete(chunk_id)

        with self._lock:
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                return deleted
            if chunk.document_id in self.documents:
                self.documents[chunk.document_id].remove_chunk(chunk_id)
            del self.chunks[chunk_id]
            return deleted

    def add_document(self, document: model.Document) -> None:
        with self._lock:
            if document.id in self.documents:
                raise ValueError(document.id)
            if document.library_id not in self.libraries:
                raise ValueError(document.library_id)
            self.documents[document.id] = document
            self.libraries[document.library_id].add_document(document.id)

    def get_document(self, document_id: str) -> Optional[model.Document]:
        with self._lock:
            return self.documents.get(document_id)

    def list_documents(self, library_id: Optional[str] = None) -> List[model.Document]:
        with self._lock:
            docs = list(self.documents.values())
            if library_id:
                docs = [d for d in docs if d.library_id == library_id]
            return docs

    def update_document_name(self, document_id: str, new_name: str) -> model.Document:
        with self._lock:
            doc = self.documents.get(document_id)
            if not doc:
                raise KeyError(document_id)
            doc.update_name(new_name)
            return doc

    def update_document_metadata(self, document_id: str, metadata: Dict[str, Any]) -> model.Document:
        with self._lock:
            doc = self.documents.get(document_id)
            if not doc:
                raise KeyError(document_id)
            doc.update_metadata(metadata)
            return doc

    def delete_document(self, document_id: str, cascade: bool = True) -> bool:
        with self._lock:
            doc = self.documents.get(document_id)
            if not doc:
                return False
            chunk_ids = list(doc.get_chunks())

        if cascade:
            for cid in chunk_ids:
                self.delete_chunk(cid)

        with self._lock:
            if doc.library_id in self.libraries:
                self.libraries[doc.library_id].remove_document(document_id)
            if document_id in self.documents:
                del self.documents[document_id]
                return True
            return False

    def add_library(self, library: model.Library) -> None:
        with self._lock:
            if library.id in self.libraries:
                raise ValueError(library.id)
            self.libraries[library.id] = library

    def get_library(self, library_id: str) -> Optional[model.Library]:
        with self._lock:
            return self.libraries.get(library_id)

    def list_libraries(self) -> List[model.Library]:
        with self._lock:
            return list(self.libraries.values())

    def update_library_name(self, library_id: str, new_name: str) -> model.Library:
        with self._lock:
            lib = self.libraries.get(library_id)
            if not lib:
                raise KeyError(library_id)
            lib.update_name(new_name)
            return lib

    def update_library_metadata(self, library_id: str, metadata: Dict[str, Any]) -> model.Library:
        with self._lock:
            lib = self.libraries.get(library_id)
            if not lib:
                raise KeyError(library_id)
            lib.update_metadata(metadata)
            return lib

    def delete_library(self, library_id: str, cascade: bool = True) -> bool:
        with self._lock:
            lib = self.libraries.get(library_id)
            if not lib:
                return False
            doc_ids = list(lib.get_documents())

        if cascade:
            for did in doc_ids:
                self.delete_document(did, cascade=True)

        with self._lock:
            if library_id in self.libraries:
                del self.libraries[library_id]
                return True
            return False

    def search(
        self,
        query_text: str,
        library_id: Optional[str] = None,
        k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **index_kwargs
    ) -> List[Tuple[str, float]]:
        client = model.get_embedding_client()
        query_embedding = client.get_embedding(query_text)

        with self._lock:
            if library_id:
                docs_in_lib = {did for did, doc in self.documents.items() if doc.library_id == library_id}
                chunks_map = {cid: c for cid, c in self.chunks.items() if c.document_id in docs_in_lib}
            else:
                chunks_map = dict(self.chunks)

        return self.index.search(
            query_embedding=query_embedding,
            chunks_map=chunks_map,
            k=k,
            metadata_filter=metadata_filter,
            **index_kwargs
        )

    def to_snapshot(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            return {
                "libraries": [l.to_dict() for l in self.libraries.values()],
                "documents": [d.to_dict() for d in self.documents.values()],
                "chunks": [c.to_dict() for c in self.chunks.values()],
            }

    def load_from_snapshot(self, snapshot: Dict[str, List[Dict[str, Any]]]) -> None:
        with self._lock:
            self.chunks.clear()
            self.documents.clear()
            self.libraries.clear()
            if hasattr(self.index, "clear"):
                self.index.clear()

        for lib_data in snapshot.get("libraries", []):
            if hasattr(model.Library, "from_dict"):
                lib = model.Library.from_dict(lib_data)
            else:
                lib = model.Library(lib_data.get("name", ""), lib_data.get("metadata", {}), lib_data.get("id"))
            with self._lock:
                self.libraries[lib.id] = lib

        for doc_data in snapshot.get("documents", []):
            if hasattr(model.Document, "from_dict"):
                doc = model.Document.from_dict(doc_data)
            else:
                doc = model.Document(doc_data.get("name", ""), doc_data.get("library_id", ""), doc_data.get("metadata", {}), doc_data.get("id"))
            with self._lock:
                self.documents[doc.id] = doc

        pairs: List[Tuple[str, List[float]]] = []
        for cdata in snapshot.get("chunks", []):
            if hasattr(model.Chunk, "from_dict"):
                chunk = model.Chunk.from_dict(cdata)
            else:
                chunk = model.Chunk(cdata.get("text", ""), cdata.get("document_id", ""), cdata.get("id"), cdata.get("metadata", {}))

            emb = cdata.get("embedding")
            if emb is not None:
                chunk.update_embedding(list(emb))

            with self._lock:
                self.chunks[chunk.id] = chunk
                if chunk.document_id in self.documents:
                    self.documents[chunk.document_id].add_chunk(chunk.id)

            emb_now = chunk.get_embedding()
            if emb_now is not None:
                pairs.append((chunk.id, emb_now))

        for cid, emb in pairs:
            self._index_insert(cid, emb)

    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "num_libraries": len(self.libraries),
                "num_documents": len(self.documents),
                "num_chunks": len(self.chunks),
            }