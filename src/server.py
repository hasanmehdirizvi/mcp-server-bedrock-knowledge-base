"""MCP Server exposing Amazon Bedrock Knowledge Bases as tools for AI agents.

This server implements the Model Context Protocol (MCP) to provide AI agents
with structured access to enterprise knowledge stored in Amazon Bedrock
Knowledge Bases. It supports semantic search, metadata filtering, and
RAG (Retrieval-Augmented Generation) through three primary tools.

Usage:
    Run directly:
        $ python -m src.server

    Or via the installed entry point:
        $ mcp-server-bedrock-kb

    Configure in Claude Desktop:
        See examples/claude_desktop_config.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from .auth import AuthenticationError, BedrockAuth
from .bedrock_client import (
    BedrockKnowledgeBaseClient,
    BedrockKnowledgeBaseError,
    KnowledgeBaseNotFoundError,
)
from .config import ServerConfig
from .models import (
    GetDocumentMetadataRequest,
    ListKnowledgeBasesRequest,
    MetadataFilter,
    QueryRequest,
    RetrievalFilter,
)

logger = logging.getLogger(__name__)

# Initialize server instance
server = Server("bedrock-knowledge-base")

# Module-level state initialized in main()
_config: ServerConfig | None = None
_client: BedrockKnowledgeBaseClient | None = None


def _get_client() -> BedrockKnowledgeBaseClient:
    """Get the initialized Bedrock client, raising if not configured."""
    if _client is None:
        raise RuntimeError("Server not initialized. Call main() first.")
    return _client


def _get_config() -> ServerConfig:
    """Get the server configuration."""
    if _config is None:
        raise RuntimeError("Server not initialized. Call main() first.")
    return _config


# --- Tool Definitions ---


@server.tool()
async def query_knowledge_base(
    knowledge_base_id: str,
    query: str,
    number_of_results: int | None = None,
    search_type: str | None = None,
    generate_response: bool = True,
    filter_and: list[dict[str, Any]] | None = None,
    filter_or: list[dict[str, Any]] | None = None,
) -> list[TextContent]:
    """Query an Amazon Bedrock Knowledge Base with natural language.

    Performs semantic search against the specified knowledge base and optionally
    generates a synthesized answer using a foundation model (RAG pattern).

    Args:
        knowledge_base_id: The ID of the knowledge base to query.
        query: Natural language question or search query.
        number_of_results: Number of passages to retrieve (1-100, default: 5).
        search_type: Search strategy - "HYBRID" (semantic + keyword) or "SEMANTIC".
        generate_response: If True, returns a model-generated answer with citations.
                          If False, returns raw retrieved passages with scores.
        filter_and: AND metadata filters. Each dict has keys: "key", "value", "operator".
                    Example: [{"key": "department", "value": "claims", "operator": "equals"}]
        filter_or: OR metadata filters. Same format as filter_and.
                   Cannot be used simultaneously with filter_and.

    Returns:
        List of TextContent with the query results formatted as structured text.
    """
    client = _get_client()

    # Build filter if provided
    retrieval_filter: RetrievalFilter | None = None
    if filter_and:
        retrieval_filter = RetrievalFilter(
            and_conditions=[MetadataFilter(**f) for f in filter_and]
        )
    elif filter_or:
        retrieval_filter = RetrievalFilter(
            or_conditions=[MetadataFilter(**f) for f in filter_or]
        )

    request = QueryRequest(
        knowledge_base_id=knowledge_base_id,
        query=query,
        number_of_results=number_of_results,
        search_type=search_type,
        generate_response=generate_response,
        filter=retrieval_filter,
    )

    try:
        response = await client.query(request)

        # Format response as structured text
        parts: list[str] = []

        if response.answer:
            parts.append(f"## Answer\n\n{response.answer}")

        if response.passages:
            parts.append(f"\n## Retrieved Passages ({len(response.passages)} results)\n")
            for i, passage in enumerate(response.passages, 1):
                score_str = f" (score: {passage.score:.4f})" if passage.score else ""
                source_str = f"\n   Source: {passage.source_uri}" if passage.source_uri else ""
                parts.append(
                    f"### Passage {i}{score_str}{source_str}\n\n{passage.content}\n"
                )

        if response.citation_count:
            parts.append(f"\n---\n*{response.citation_count} citation(s) referenced*")

        result_text = "\n".join(parts) if parts else "No results found for the given query."

        return [TextContent(type="text", text=result_text)]

    except KnowledgeBaseNotFoundError as e:
        return [TextContent(type="text", text=f"Error: {e}")]
    except BedrockKnowledgeBaseError as e:
        logger.error("Knowledge base query failed: %s", e)
        return [TextContent(type="text", text=f"Error querying knowledge base: {e}")]


@server.tool()
async def list_knowledge_bases(
    max_results: int = 10,
    next_token: str | None = None,
) -> list[TextContent]:
    """List available Amazon Bedrock Knowledge Bases.

    Returns information about knowledge bases accessible to this server,
    including their IDs, names, descriptions, and current status.

    Args:
        max_results: Maximum number of knowledge bases to return (1-100).
        next_token: Pagination token from a previous response.

    Returns:
        List of TextContent with formatted knowledge base information.
    """
    client = _get_client()

    try:
        response = await client.list_knowledge_bases(
            max_results=max_results,
            next_token=next_token,
        )

        if not response.knowledge_bases:
            return [TextContent(type="text", text="No knowledge bases found.")]

        parts: list[str] = ["## Available Knowledge Bases\n"]
        for kb in response.knowledge_bases:
            description = f"\n   Description: {kb.description}" if kb.description else ""
            updated = f"\n   Last Updated: {kb.updated_at.isoformat()}" if kb.updated_at else ""
            parts.append(
                f"- **{kb.name}** (`{kb.knowledge_base_id}`)\n"
                f"   Status: {kb.status}{description}{updated}\n"
            )

        if response.next_token:
            parts.append(f"\n*More results available. Use next_token: `{response.next_token}`*")

        return [TextContent(type="text", text="\n".join(parts))]

    except BedrockKnowledgeBaseError as e:
        logger.error("Failed to list knowledge bases: %s", e)
        return [TextContent(type="text", text=f"Error listing knowledge bases: {e}")]


@server.tool()
async def get_document_metadata(
    knowledge_base_id: str,
    data_source_id: str,
    document_uri: str | None = None,
) -> list[TextContent]:
    """Get metadata about documents in a Bedrock Knowledge Base data source.

    Retrieves ingestion status and metadata for documents within a specific
    data source of a knowledge base. Useful for verifying document availability
    and checking sync status.

    Args:
        knowledge_base_id: The knowledge base ID containing the data source.
        data_source_id: The data source ID to inspect.
        document_uri: Optional S3 URI to filter for a specific document.

    Returns:
        List of TextContent with document metadata information.
    """
    client = _get_client()

    try:
        response = await client.get_document_metadata(
            knowledge_base_id=knowledge_base_id,
            data_source_id=data_source_id,
            document_uri=document_uri,
        )

        if not response.documents:
            return [TextContent(type="text", text="No documents found for the given criteria.")]

        parts: list[str] = [
            f"## Document Metadata\n"
            f"Knowledge Base: `{response.knowledge_base_id}`\n"
            f"Data Source: `{response.data_source_id}`\n"
        ]

        for doc in response.documents:
            updated = f"\n   Last Updated: {doc.last_updated.isoformat()}" if doc.last_updated else ""
            metadata_str = ""
            if doc.metadata:
                metadata_str = f"\n   Metrics: {json.dumps(doc.metadata, default=str)}"
            parts.append(
                f"- **{doc.document_uri}**\n"
                f"   Status: {doc.status}{updated}{metadata_str}\n"
            )

        return [TextContent(type="text", text="\n".join(parts))]

    except BedrockKnowledgeBaseError as e:
        logger.error("Failed to get document metadata: %s", e)
        return [TextContent(type="text", text=f"Error retrieving document metadata: {e}")]


# --- Server Lifecycle ---


def main() -> None:
    """Initialize and run the MCP server.

    Loads configuration from environment, validates AWS credentials,
    and starts the MCP server on stdio transport.
    """
    global _config, _client

    # Load configuration
    _config = ServerConfig.from_env()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, _config.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # MCP uses stdout for protocol, logs go to stderr
    )

    logger.info(
        "Starting MCP server '%s' in region %s",
        _config.server_name,
        _config.aws_region,
    )

    if _config.knowledge_base_ids:
        logger.info("Configured knowledge bases: %s", _config.knowledge_base_ids)
    else:
        logger.info("No specific KB IDs configured; all accessible KBs will be available")

    # Initialize authentication and client
    try:
        auth = BedrockAuth(_config)
        _client = BedrockKnowledgeBaseClient(_config, auth)
        logger.info("Bedrock client initialized successfully")
    except AuthenticationError as e:
        logger.error("Authentication failed: %s", e)
        sys.exit(1)

    # Run server with stdio transport
    async def run() -> None:
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(run())


if __name__ == "__main__":
    main()
