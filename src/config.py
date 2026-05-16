"""Configuration management for the Bedrock Knowledge Base MCP Server.

Supports environment variables and programmatic configuration for
knowledge base IDs, AWS region, model selection, and retrieval parameters.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class RetrievalConfig(BaseSettings):
    """Configuration for knowledge base retrieval behavior."""

    number_of_results: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results to return from retrieval",
    )
    search_type: Literal["HYBRID", "SEMANTIC"] = Field(
        default="HYBRID",
        description="Search type: HYBRID (semantic + keyword) or SEMANTIC only",
    )
    override_search_type: bool = Field(
        default=False,
        description="Whether to override the KB's default search type configuration",
    )

    model_config = {"env_prefix": "BEDROCK_KB_RETRIEVAL_"}


class ServerConfig(BaseSettings):
    """Main server configuration loaded from environment variables.

    Environment Variables:
        AWS_REGION: AWS region for Bedrock API calls (default: us-west-2)
        AWS_PROFILE: AWS profile name for credential resolution
        BEDROCK_KB_IDS: Comma-separated list of knowledge base IDs to expose
        BEDROCK_KB_MODEL_ID: Foundation model ARN for RetrieveAndGenerate
        BEDROCK_KB_SERVER_NAME: MCP server name identifier
        BEDROCK_KB_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
    """

    aws_region: str = Field(
        default="us-west-2",
        description="AWS region for Bedrock Knowledge Base API calls",
    )
    aws_profile: str | None = Field(
        default=None,
        description="AWS profile for credential resolution; uses default chain if unset",
    )
    knowledge_base_ids: list[str] = Field(
        default_factory=list,
        description="List of knowledge base IDs this server can query",
    )
    model_id: str = Field(
        default="arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-sonnet-4-20250514",
        description="Foundation model ARN for RetrieveAndGenerate responses",
    )
    server_name: str = Field(
        default="bedrock-knowledge-base",
        description="MCP server name used in protocol handshake",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity level",
    )
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    model_config = {
        "env_prefix": "BEDROCK_KB_",
        "env_nested_delimiter": "__",
    }

    @classmethod
    def from_env(cls) -> ServerConfig:
        """Load configuration from environment variables.

        Knowledge base IDs should be provided as a comma-separated string
        in the BEDROCK_KB_KNOWLEDGE_BASE_IDS environment variable.
        """
        import os

        config = cls()

        # Handle comma-separated KB IDs from env
        kb_ids_env = os.environ.get("BEDROCK_KB_IDS", "")
        if kb_ids_env and not config.knowledge_base_ids:
            config.knowledge_base_ids = [
                kb_id.strip() for kb_id in kb_ids_env.split(",") if kb_id.strip()
            ]

        return config
