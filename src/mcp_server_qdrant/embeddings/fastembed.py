import asyncio
from typing import List
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.logger import get_logger
from fastembed import TextEmbedding
from fastembed.common.model_description import DenseModelDescription
import logging
import os
import sys

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        # File handler
        log_file = os.path.join(LOG_DIR, f"{name.replace('.', '_')}.log")
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)

class FastEmbedProvider(EmbeddingProvider):
    """
    FastEmbed implementation of the embedding provider.
    :param model_name: The name of the FastEmbed model to use.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_model = TextEmbedding(model_name)

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        logger.debug(f"embed_documents called. Documents: {documents}")
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.passage_embed(documents))
        )
        logger.debug(f"embed_documents result (first 5 values): {[emb[:5] for emb in embeddings]}")
        return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        logger.debug(f"embed_query called. Query: {query}")
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.query_embed([query]))
        )
        result = embeddings[0].tolist()
        logger.debug(f"embed_query result (first 5 values): {result[:5]}")
        return result


    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        model_description: DenseModelDescription = (
            self.embedding_model._get_model_description(self.model_name)
        )
        return model_description.dim
