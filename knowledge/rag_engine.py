import hashlib
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunks.append(" ".join(words[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


class RAGEngine:
    def __init__(
        self,
        documents_path: str,
        chroma_path: str,
        top_k: int = 3,
        min_similarity: float = 0.4,
    ):
        self._docs_path = Path(documents_path)
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._client = chromadb.PersistentClient(path=chroma_path)
        # cosine space ensures distances match cosine similarity as specified
        self._collection = self._client.get_or_create_collection(
            "vedaai_knowledge",
            metadata={"hnsw:space": "cosine"},
        )

    async def index_documents(self) -> None:
        """Index all .txt and .md files. Runs blocking I/O in a thread executor."""
        import asyncio
        await asyncio.to_thread(self._index_documents_sync)

    def _index_documents_sync(self) -> None:
        """Synchronous implementation — called via asyncio.to_thread."""
        for file_path in self._docs_path.glob("**/*"):
            if file_path.suffix not in (".txt", ".md"):
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                continue
            for i, chunk in enumerate(_chunk_text(text)):
                chunk_id = f"{file_path.relative_to(self._docs_path)}::{i}"
                chunk_hash = hashlib.sha256(chunk.encode()).hexdigest()[:16]
                existing = self._collection.get(ids=[chunk_id])
                if existing["ids"] and existing["metadatas"][0].get("hash") == chunk_hash:
                    continue
                embedding = self._model.encode(chunk).tolist()
                self._collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "filename": file_path.name,
                        "chunk": i,
                        "hash": chunk_hash,
                    }],
                )

    def _search_sync(self, query: str) -> list[str]:
        """Blocking search — encode + ChromaDB query. Called via asyncio.to_thread."""
        n = min(self._top_k, self._collection.count())
        query_embedding = self._model.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "distances"],
        )
        chunks = []
        for doc, distance in zip(results["documents"][0], results["distances"][0]):
            # ChromaDB cosine space returns distance in [0, 2]; similarity = 1 - distance/2
            similarity = 1.0 - distance / 2.0
            if similarity >= self._min_similarity:
                chunks.append(doc)
        return chunks

    async def search(self, query: str) -> list[str]:
        """Returns text chunks with cosine similarity >= min_similarity."""
        if self._collection.count() == 0:
            return []
        import asyncio
        return await asyncio.to_thread(self._search_sync, query)

    async def search_as_string(self, query: str) -> str:
        """Returns chunks joined as a single string for prompt injection."""
        chunks = await self.search(query)
        return "\n\n".join(chunks) if chunks else ""
