# Her modül kendi log dosyasına yazar (logs klasörü altında)
import asyncio
from typing import List
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.logger import get_logger
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

class SentenceTransformersProvider(EmbeddingProvider):
    """
    SentenceTransformers implementation of the embedding provider.
    :param model_name: The name of the SentenceTransformers model to use.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.logger = get_logger(__name__)

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        logger.debug(f"embed_documents called. Documents: {documents}")
        # Run in a thread pool since SentenceTransformers is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.encode(documents))
        )
        logger.debug(f"embed_documents result (first 5 values): {[emb[:5] for emb in embeddings]}")
        return [embedding.tolist() for embedding in embeddings]
    
    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        logger.debug(f"embed_query called. Query: {query}")
        # Run in a thread pool since SentenceTransformers is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.encode([query]))
        )
        result = embeddings[0].tolist()
        logger.debug(f"embed_query result (first 5 values): {result[:5]}")
        return result
    

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        return self.embedding_model.get_sentence_embedding_dimension()