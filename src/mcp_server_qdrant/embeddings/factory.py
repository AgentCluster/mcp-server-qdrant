from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import EmbeddingProviderSettings


def create_embedding_provider(settings: EmbeddingProviderSettings) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    :param settings: The settings for the embedding provider.
    :return: An instance of the specified embedding provider.
    """
    if settings.provider_type == EmbeddingProviderType.FASTEMBED:
        from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider

        return FastEmbedProvider(settings.model_name)
    elif settings.provider_type == EmbeddingProviderType.SENTENCE_TRANSFORMERS:
        from mcp_server_qdrant.embeddings.sentence_transformens import SentenceTransformersProvider

        return SentenceTransformersProvider(settings.model_name)
    elif settings.provider_type == EmbeddingProviderType.GEMINI_TRANSFORMER:
        from mcp_server_qdrant.embeddings.gemini_transformer import GeminiTransformerProvider

        return GeminiTransformerProvider(settings.model_name)
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")
