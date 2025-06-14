import os
import asyncio
from typing import List
from google import genai
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.logger import get_logger


logger = get_logger(__name__)


class GeminiTransformerProvider(EmbeddingProvider):
    """
    GeminiTransformer implementation of the embedding provider.
    :param model_name: The name of the GeminiTransformer model to use.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.logger = get_logger(__name__)

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        logger.debug(f"embed_documents called. Documents: {documents}")
        loop = asyncio.get_event_loop()
        def embed():
            result = self.client.models.embed_content(model=self.model_name, contents=documents)
            return [embedding.values for embedding in result.embeddings]
        embeddings = await loop.run_in_executor(None, embed)
        logger.debug(f"embed_documents result (first 5 values): {[emb[:5] for emb in embeddings]}")
        return embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        logger.debug(f"embed_query called. Query: {query}")
        loop = asyncio.get_event_loop()
        def embed():
            result = self.client.models.embed_content(model=self.model_name, contents=[query])
            return result.embeddings[0].values
        embedding = await loop.run_in_executor(None, embed)
        return embedding
    
    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        return 768