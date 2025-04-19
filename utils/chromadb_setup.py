import chromadb
from .config import CHROMA_PATH, COLLECTION_NAME # Import from config.py

def initialize_chromadb_collection():
    """Initializes ChromaDB client and returns the collection."""
    try:
        # Use PersistentClient for data to survive script restarts
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

        # Get or create a collection
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"} # Use cosine distance
        )
        print(f"ChromaDB collection '{COLLECTION_NAME}' loaded/created from path: {CHROMA_PATH}")
        print(f"Number of items currently in collection: {collection.count()}")
        return collection
    except Exception as e:
        print(f"Error initializing ChromaDB collection: {e}")
        raise # Re-raise the exception

