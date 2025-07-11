from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType

DEFAULT_TOOL_STORE_DESCRIPTION = (
    "Keep the memory for later use, when you are asked to remember something."
)
DEFAULT_TOOL_FIND_DESCRIPTION = (
    "Look up memories in Qdrant. Use this tool when you need to: \n"
    " - Find memories by their content \n"
    " - Access memories for further analysis \n"
    " - Get some personal information about the user"
)
DEFAULT_TOOL_FIND_BY_METADATA_DESCRIPTION = (
    "Find specific vectors in Qdrant by metadata key-value pairs. Use this tool when you need to: \n"
    " - Filter results by specific metadata fields like 'mahkeme', 'durum', 'karar_no', etc. \n"
    " - Find all documents that have a specific metadata value \n"
    " - Search for documents by their properties rather than content"
)


class ToolSettings(BaseSettings):
    """
    Configuration for all the tools.
    """

    tool_store_description: str = Field(
        default=DEFAULT_TOOL_STORE_DESCRIPTION,
        validation_alias="TOOL_STORE_DESCRIPTION",
    )
    tool_find_description: str = Field(
        default=DEFAULT_TOOL_FIND_DESCRIPTION,
        validation_alias="TOOL_FIND_DESCRIPTION",
    )
    tool_find_by_metadata_description: str = Field(
        default=DEFAULT_TOOL_FIND_BY_METADATA_DESCRIPTION,
        validation_alias="TOOL_FIND_BY_METADATA_DESCRIPTION",
    )


class EmbeddingProviderSettings(BaseSettings):
    """
    Configuration for the embedding provider.
    """

    provider_type: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.FASTEMBED,
        validation_alias="EMBEDDING_PROVIDER",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )


class QdrantSettings(BaseSettings):
    """
    Configuration for the Qdrant connector.
    """

    location: Optional[str] = Field(default=None, validation_alias="QDRANT_URL")
    api_key: Optional[str] = Field(default=None, validation_alias="QDRANT_API_KEY")
    collection_name: Optional[str] = Field(
        default=None, validation_alias="COLLECTION_NAME"
    )
    local_path: Optional[str] = Field(
        default=None, validation_alias="QDRANT_LOCAL_PATH"
    )
    search_limit: int = Field(default=10, validation_alias="QDRANT_SEARCH_LIMIT")
    read_only: bool = Field(default=False, validation_alias="QDRANT_READ_ONLY")
    timeout: int = Field(default=30, validation_alias="QDRANT_TIMEOUT")
    output_format: str = Field(default="formatted", validation_alias="QDRANT_OUTPUT_FORMAT")
