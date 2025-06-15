import uuid
import os
from typing import Any, Dict, Optional
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.logger import get_logger

logger = get_logger(__name__)

Metadata = Dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    source_id: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    docAuthor: Optional[str] = None
    description: Optional[str] = None
    docSource: Optional[str] = None
    published: Optional[str] = None
    wordCount: Optional[int] = None
    tokenCountEstimate: Optional[str] = None
    text: Optional[str] = None
    metadata: Optional[Metadata] = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: Optional[str],
        qdrant_api_key: Optional[str],
        collection_name: Optional[str],
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: Optional[str] = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path, timeout=int(os.getenv("QDRANT_TIMEOUT", "30"))
        )

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: Optional[str] = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        logger.debug(f"store called. Entry: {entry}, Collection: {collection_name}")
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        # Embed the document
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.
        embeddings = await self._embedding_provider.embed_documents([entry.content])
        logger.debug(f"store embedding result (first 5 values): {embeddings[0][:5] if embeddings else None}")

        payload = {
            "source_id": entry.source_id,
            "url": entry.url,
            "title": entry.title,
            "docAuthor": entry.docAuthor,
            "description": entry.description,
            "docSource": entry.docSource,
            "published": entry.published,
            "wordCount": entry.wordCount,
            "tokenCountEstimate": entry.tokenCountEstimate,
            "text": entry.text,
        }
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    #vector={vector_name: embeddings[0]},
                    payload=payload,
                )
            ],
        )
        logger.debug(f"store completed. Collection: {collection_name}, Payload: {payload}")

    async def search(
        self, query: str, *, collection_name: Optional[str] = None, limit: int = 10
    ) -> list[Dict[str, Any]]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :return: A list of entries found.
        """
        logger.debug(f"search called. Query: {query}, Collection: {collection_name}, Limit: {limit}")
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            logger.debug(f"search: Collection not found: {collection_name}")
            return []

        # Embed the query
        query_vector = await self._embedding_provider.embed_query(query)
        logger.debug(f"query_vector type: {type(query_vector)}")
        logger.debug(f"query_vector value: {query_vector}")
        logger.debug(f"search embedding result (first 5 values): {query_vector[:5] if query_vector else None}")

        # Search in Qdrant
        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
        )

        logger.debug(f"search results: {len(search_results.points)} found.")
        return [
            {
                "id": result.payload.get("id"),
                "karar_no": result.payload.get("karar_no"),
                "karar_tarihi": result.payload.get("karar_tarihi"),
                "mahkeme": result.payload.get("daire"),
                "esas_no": result.payload.get("esas_no"),
                "durum": result.payload.get("durum"),
                "imported_at": result.payload.get("imported_at"),
                "text": result.payload.get("text")
            }
            for result in search_results.points
        ]

    async def search_by_metadata(
        self, metadata_key: str, metadata_value: str, *, collection_name: Optional[str] = None, limit: int = 10
    ) -> list[Dict[str, Any]]:
        """
        Find points in the Qdrant collection by metadata key-value pair. If there are no entries found, an empty list is returned.
        :param metadata_key: The metadata key to search for.
        :param metadata_value: The metadata value to match.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                               the default collection is used.
        :param limit: The maximum number of entries to return.
        :return: A list of entries found.
        """
        logger.debug(f"search_by_metadata called. Key: {metadata_key}, Value: {metadata_value}, Collection: {collection_name}, Limit: {limit}")
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            logger.debug(f"search_by_metadata: Collection not found: {collection_name}")
            return []

        # Create filter condition for metadata search
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key=metadata_key,
                    match=models.MatchValue(value=metadata_value)
                )
            ]
        )

        # Search in Qdrant with filter
        search_results = await self._client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_condition,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        logger.debug(f"search_by_metadata results: {len(search_results[0])} found.")
        return [
            {
                "id": result.payload.get("id"),
                "karar_no": result.payload.get("karar_no"),
                "karar_tarihi": result.payload.get("karar_tarihi"),
                "mahkeme": result.payload.get("daire"),
                "esas_no": result.payload.get("esas_no"),
                "durum": result.payload.get("durum"),
                "imported_at": result.payload.get("imported_at"),
                "text": result.payload.get("text")
            }
            for result in search_results[0]
        ]

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()

            # Use the vector name as defined in the embedding provider
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
            )
