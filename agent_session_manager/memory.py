"""
ChromaDB Semantic Memory Layer for Agent Session Manager.

Provides semantic storage and retrieval of agent context using
local sentence-transformers embeddings — no cloud dependencies.
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import List, Dict, Optional, Any
import hashlib


class ChromaDBMemory:
    """ChromaDB-based semantic memory using sentence-transformers embeddings."""

    def __init__(
        self,
        collection_name: str = "agent_memory",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize ChromaDB memory.

        Args:
            collection_name: Name of the ChromaDB collection to use
            persist_directory: Path to the ChromaDB persistence directory
            embedding_model: Sentence-transformers model name for embeddings
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Use sentence-transformers for local embeddings
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    # ------------------------------------------------------------------ #
    # Primary API used by SessionManager                                   #
    # ------------------------------------------------------------------ #

    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Add a text document to semantic memory.

        Args:
            text: Text content to store and embed
            metadata: Optional metadata dictionary
            session_id: Session this document belongs to

        Returns:
            Document ID assigned
        """
        # Generate a stable unique ID from session + content
        id_source = f"{session_id or ''}_{text}"
        doc_id = hashlib.md5(id_source.encode()).hexdigest()

        # Build metadata, always stamping session_id
        doc_metadata: Dict[str, Any] = {}
        if session_id:
            doc_metadata["session_id"] = session_id
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    doc_metadata[key] = value
                else:
                    doc_metadata[key] = str(value)

        # Upsert to avoid duplicate-key errors on re-indexing
        self.collection.upsert(
            documents=[text],
            metadatas=[doc_metadata],
            ids=[doc_id]
        )
        return doc_id

    def search_similar(
        self,
        query_text: str,
        n_results: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar documents.

        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            filter_dict: Additional ChromaDB metadata filters
            session_id: Optional session ID to restrict results to

        Returns:
            List of result dicts with keys: text, metadata, distance, id
        """
        # Build where filter
        where_filter: Optional[Dict[str, Any]] = None
        if session_id and filter_dict:
            where_filter = {"$and": [{"session_id": session_id}, filter_dict]}
        elif session_id:
            where_filter = {"session_id": session_id}
        elif filter_dict:
            where_filter = filter_dict

        # Guard against querying more results than documents in collection
        count = self.collection.count()
        if count == 0:
            return []
        n_results = min(n_results, count)

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )

        formatted: List[Dict[str, Any]] = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                    "id": results["ids"][0][i] if results["ids"] else None,
                })
        return formatted

    def delete_session_documents(self, session_id: str) -> int:
        """
        Delete all documents belonging to a session.

        Args:
            session_id: Session whose documents should be removed

        Returns:
            Number of documents deleted
        """
        try:
            results = self.collection.get(where={"session_id": session_id})
            ids = results.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def count_documents(self, session_id: Optional[str] = None) -> int:
        """
        Count documents, optionally scoped to a session.

        Args:
            session_id: If provided, count only this session's documents

        Returns:
            Document count
        """
        if session_id is None:
            return self.collection.count()
        try:
            results = self.collection.get(where={"session_id": session_id})
            return len(results.get("ids", []))
        except Exception:
            return 0

    # ------------------------------------------------------------------ #
    # Legacy / convenience aliases                                         #
    # ------------------------------------------------------------------ #

    def add(
        self,
        session_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Legacy alias for add_document (session_id first)."""
        return self.add_document(text=text, metadata=metadata, session_id=session_id)

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Legacy alias for search_similar."""
        return self.search_similar(
            query_text=query,
            n_results=n_results,
            session_id=filter_session_id
        )


# ------------------------------------------------------------------ #
# Self-test (invoked by README instructions)                           #
# ------------------------------------------------------------------ #

def test_memory():
    """Test the ChromaDB memory layer."""
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    try:
        print("Initializing ChromaDBMemory...")
        memory = ChromaDBMemory(
            collection_name="test_collection",
            persist_directory=temp_dir,
            embedding_model="all-MiniLM-L6-v2"
        )

        print("\nAdding documents...")
        memory.add_document(
            text="The agent initialized the database connection",
            metadata={"type": "step", "step_number": 1},
            session_id="test-session"
        )
        memory.add_document(
            text="Authentication middleware was configured with JWT tokens",
            metadata={"type": "step", "step_number": 2},
            session_id="test-session"
        )
        memory.add_document(
            text="Unit tests achieved 95% code coverage",
            metadata={"type": "step", "step_number": 3},
            session_id="test-session"
        )
        print("  3 documents added")

        print("\nSearching for similar documents...")
        results = memory.search_similar(
            query_text="database initialization",
            n_results=2,
            session_id="test-session"
        )
        print(f"  Found {len(results)} results")
        for r in results:
            print(f"    - {r['text'][:60]}... (distance: {r['distance']:.4f})")

        print("\nCounting documents...")
        total = memory.count_documents()
        session_count = memory.count_documents("test-session")
        print(f"  Total: {total}, Session: {session_count}")

        print("\nDeleting session documents...")
        deleted = memory.delete_session_documents("test-session")
        print(f"  Deleted: {deleted}")
        print(f"  Remaining: {memory.count_documents()}")

        print("\n✅ All memory tests passed!")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_memory()
