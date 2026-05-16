"""Example client usage for the Bedrock Knowledge Base MCP Server.

Demonstrates how to programmatically connect to the MCP server
and invoke its tools using the MCP Python SDK client.
"""

from __future__ import annotations

import asyncio
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    """Connect to the Bedrock KB MCP server and execute example queries."""

    # Configure connection to the MCP server
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.server"],
        env={
            "AWS_REGION": "us-west-2",
            "AWS_PROFILE": "bedrock",
            "BEDROCK_KB_IDS": "YOUR_KB_ID_HERE",
            "BEDROCK_KB_LOG_LEVEL": "DEBUG",
        },
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the MCP session
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print("Available tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description[:80]}...")
            print()

            # --- Example 1: List Knowledge Bases ---
            print("=" * 60)
            print("Example 1: Listing available knowledge bases")
            print("=" * 60)

            result = await session.call_tool("list_knowledge_bases", {"max_results": 5})
            for content in result.content:
                print(content.text)
            print()

            # --- Example 2: Query with RAG (RetrieveAndGenerate) ---
            print("=" * 60)
            print("Example 2: RAG query with generated answer")
            print("=" * 60)

            result = await session.call_tool(
                "query_knowledge_base",
                {
                    "knowledge_base_id": "YOUR_KB_ID_HERE",
                    "query": "What are the coverage limits for commercial auto insurance?",
                    "generate_response": True,
                    "number_of_results": 5,
                },
            )
            for content in result.content:
                print(content.text)
            print()

            # --- Example 3: Retrieve raw passages (no generation) ---
            print("=" * 60)
            print("Example 3: Raw passage retrieval without generation")
            print("=" * 60)

            result = await session.call_tool(
                "query_knowledge_base",
                {
                    "knowledge_base_id": "YOUR_KB_ID_HERE",
                    "query": "Claims processing workflow for property damage",
                    "generate_response": False,
                    "number_of_results": 3,
                    "search_type": "SEMANTIC",
                },
            )
            for content in result.content:
                print(content.text)
            print()

            # --- Example 4: Query with metadata filtering ---
            print("=" * 60)
            print("Example 4: Query with metadata filters")
            print("=" * 60)

            result = await session.call_tool(
                "query_knowledge_base",
                {
                    "knowledge_base_id": "YOUR_KB_ID_HERE",
                    "query": "What is the deductible for flood damage?",
                    "generate_response": True,
                    "filter_and": [
                        {"key": "document_type", "value": "policy", "operator": "equals"},
                        {"key": "state", "value": "Florida", "operator": "equals"},
                    ],
                },
            )
            for content in result.content:
                print(content.text)
            print()

            # --- Example 5: Get document metadata ---
            print("=" * 60)
            print("Example 5: Document metadata inspection")
            print("=" * 60)

            result = await session.call_tool(
                "get_document_metadata",
                {
                    "knowledge_base_id": "YOUR_KB_ID_HERE",
                    "data_source_id": "YOUR_DATA_SOURCE_ID",
                },
            )
            for content in result.content:
                print(content.text)


if __name__ == "__main__":
    asyncio.run(main())
