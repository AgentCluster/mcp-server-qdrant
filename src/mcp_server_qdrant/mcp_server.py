import json
import os
from typing import Any, List
from mcp_server_qdrant.logger import get_logger

from mcp.server.fastmcp import Context, FastMCP

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = get_logger(__name__)

# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: EmbeddingProviderSettings,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):

        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "80"))
        
        # Override any existing settings with our host/port
        settings.update({
            "host": host,
            "port": port,
        })
        


        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings
        self.embedding_provider_settings = embedding_provider_settings

        self.embedding_provider = create_embedding_provider(embedding_provider_settings)
        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        """
        Register the tools in the server.
        """
        async def store(
            ctx: Context,
            information: str,
            collection_name: str,
            metadata: Metadata = None,  # type: ignore
        ) -> str:
            logger.debug(f"store tool called. information: {information}, collection_name: {collection_name}, metadata: {metadata}")
            await ctx.debug(f"Storing information {information} in Qdrant")
            entry = Entry(
                content=information,
                source_id=metadata.get("source_id") if metadata else None,
                url=metadata.get("url") if metadata else None,
                title=metadata.get("title") if metadata else None,
                docAuthor=metadata.get("docAuthor") if metadata else None,
                description=metadata.get("description") if metadata else None,
                docSource=metadata.get("docSource") if metadata else None,
                published=metadata.get("published") if metadata else None,
                wordCount=metadata.get("wordCount") if metadata else None,
                tokenCountEstimate=metadata.get("tokenCountEstimate") if metadata else None,
                text=metadata.get("text") if metadata else None,
                metadata=metadata,
            )
            await self.qdrant_connector.store(entry, collection_name=collection_name)
            logger.debug(f"store tool completed. information: {information}, collection_name: {collection_name}")
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def store_with_default_collection(
            ctx: Context,
            information: str,
            metadata: Metadata = None,  # type: ignore
        ) -> str:
            assert self.qdrant_settings.collection_name is not None
            logger.debug(f"store_with_default_collection called. information: {information}, metadata: {metadata}")
            return await store(
                ctx, information, self.qdrant_settings.collection_name, metadata
            )

        async def sanitize_input(value):
            from collections.abc import Awaitable
            if isinstance(value, Awaitable):
                return await value
            return value

        async def find(
            ctx: Context,
            query: str,
            collection_name: str,
        ) -> List[str]:
            
            logger.debug(f"find tool called. query type: {type(query)}")
            query = await sanitize_input(query)
            logger.debug(f"find tool called. query type2: {type(query)}")
            logger.debug(f"find tool called. query: {query}, collection_name: {collection_name}")
            await ctx.debug(f"Finding results for query {query}")
            if collection_name:
                await ctx.debug(
                    f"Overriding the collection name with {collection_name}"
                )
            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
            )
            if not entries:
                logger.debug(f"find tool: No result found. query: {query}")
                return [f"No information found for the query '{query}'"]
            logger.debug(f"find tool: {len(entries)} results found. query: {query}")
            return entries

        async def find_with_default_collection(
            ctx: Context,
            query: str,
        ) -> List[str]:
            assert self.qdrant_settings.collection_name is not None
            logger.debug(f"find_with_default_collection called. query: {query}")
            return await find(ctx, query, self.qdrant_settings.collection_name)

        async def find_by_metadata(
            ctx: Context,
            metadata_key: str,
            metadata_value: str,
            collection_name: str,
        ) -> List[str]:
            logger.debug(f"find_by_metadata tool called. key: {metadata_key}, value: {metadata_value}, collection_name: {collection_name}")
            await ctx.debug(f"Finding results by metadata {metadata_key}={metadata_value}")
            if collection_name:
                await ctx.debug(
                    f"Using collection: {collection_name}"
                )
            entries = await self.qdrant_connector.search_by_metadata(
                metadata_key,
                metadata_value,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
            )
            if not entries:
                logger.debug(f"find_by_metadata tool: No result found. key: {metadata_key}, value: {metadata_value}")
                return [f"No information found for metadata {metadata_key}='{metadata_value}'"]
            logger.debug(f"find_by_metadata tool: {len(entries)} results found. key: {metadata_key}, value: {metadata_value}")
            return entries

        async def find_by_metadata_with_default_collection(
            ctx: Context,
            metadata_key: str,
            metadata_value: str,
        ) -> List[str]:
            assert self.qdrant_settings.collection_name is not None
            logger.debug(f"find_by_metadata_with_default_collection called. key: {metadata_key}, value: {metadata_value}")
            return await find_by_metadata(ctx, metadata_key, metadata_value, self.qdrant_settings.collection_name)

        # Register the tools depending on the configuration

        if self.qdrant_settings.collection_name:
            self.add_tool(
                find_with_default_collection,
                name="qdrant-find",
                description=self.tool_settings.tool_find_description,
            )
            self.add_tool(
                find_by_metadata_with_default_collection,
                name="qdrant-find-by-metadata",
                description=self.tool_settings.tool_find_by_metadata_description,
            )
        else:
            self.add_tool(
                find,
                name="qdrant-find",
                description=self.tool_settings.tool_find_description,
            )
            self.add_tool(
                find_by_metadata,
                name="qdrant-find-by-metadata",
                description=self.tool_settings.tool_find_by_metadata_description,
            )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database

            if self.qdrant_settings.collection_name:
                self.add_tool(
                    store_with_default_collection,
                    name="qdrant-store",
                    description=self.tool_settings.tool_store_description,
                )
            else:
                self.add_tool(
                    store,
                    name="qdrant-store",
                    description=self.tool_settings.tool_store_description,
                )
