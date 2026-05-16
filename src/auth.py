"""IAM authentication helper for AWS Bedrock API access.

Manages boto3 session lifecycle, credential refresh, and provides
pre-configured clients for Bedrock Agent Runtime and Bedrock Agent APIs.
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    TokenRetrievalError,
)

if TYPE_CHECKING:
    from mypy_boto3_bedrock_agent import BedrockAgentClient
    from mypy_boto3_bedrock_agent_runtime import BedrockAgentRuntimeClient

from .config import ServerConfig

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Raised when AWS credential resolution or validation fails."""


class BedrockAuth:
    """Manages AWS authentication and client creation for Bedrock services.

    Creates and caches boto3 clients with proper retry configuration,
    timeouts, and user-agent identification for the MCP server.

    Args:
        config: Server configuration containing region and profile settings.
    """

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._session: boto3.Session | None = None

    @property
    def session(self) -> boto3.Session:
        """Get or create an authenticated boto3 session.

        Uses the configured AWS profile if set, otherwise falls back
        to the default credential chain (env vars, instance profile, etc.).
        """
        if self._session is None:
            try:
                session_kwargs: dict = {"region_name": self._config.aws_region}
                if self._config.aws_profile:
                    session_kwargs["profile_name"] = self._config.aws_profile

                self._session = boto3.Session(**session_kwargs)
                # Validate credentials are resolvable
                credentials = self._session.get_credentials()
                if credentials is None:
                    raise AuthenticationError(
                        "No AWS credentials found. Configure credentials via environment "
                        "variables, AWS profile, or instance metadata."
                    )

                logger.info(
                    "AWS session established in region %s",
                    self._config.aws_region,
                )
            except NoCredentialsError as e:
                raise AuthenticationError(
                    f"AWS credentials not found: {e}. "
                    "Ensure AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY are set, "
                    "or configure an AWS profile."
                ) from e
            except TokenRetrievalError as e:
                raise AuthenticationError(
                    f"Failed to retrieve SSO/session token: {e}. "
                    "Run 'aws sso login' if using SSO."
                ) from e

        return self._session

    @cached_property
    def _boto_config(self) -> BotoConfig:
        """Shared boto3 client configuration with retry and timeout settings."""
        return BotoConfig(
            region_name=self._config.aws_region,
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=5,
            read_timeout=30,
            user_agent_extra="mcp-server-bedrock-kb/0.1.0",
        )

    @cached_property
    def bedrock_agent_runtime_client(self) -> "BedrockAgentRuntimeClient":
        """Bedrock Agent Runtime client for Retrieve and RetrieveAndGenerate APIs."""
        return self.session.client(
            "bedrock-agent-runtime",
            config=self._boto_config,
        )

    @cached_property
    def bedrock_agent_client(self) -> "BedrockAgentClient":
        """Bedrock Agent client for management APIs (ListKnowledgeBases, etc.)."""
        return self.session.client(
            "bedrock-agent",
            config=self._boto_config,
        )

    def validate_access(self) -> bool:
        """Validate that credentials have access to Bedrock Knowledge Base APIs.

        Performs a lightweight ListKnowledgeBases call to verify permissions.

        Returns:
            True if access is validated successfully.

        Raises:
            AuthenticationError: If credentials lack required permissions.
        """
        try:
            self.bedrock_agent_client.list_knowledge_bases(maxResults=1)
            logger.info("Bedrock Knowledge Base access validated successfully")
            return True
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ("AccessDeniedException", "UnauthorizedAccessException"):
                raise AuthenticationError(
                    f"Insufficient permissions for Bedrock Knowledge Base APIs: {e}. "
                    "Ensure the IAM role/user has bedrock:ListKnowledgeBases, "
                    "bedrock:Retrieve, and bedrock:RetrieveAndGenerate permissions."
                ) from e
            raise

    def reset(self) -> None:
        """Reset cached session and clients, forcing credential refresh."""
        self._session = None
        # Clear cached properties
        for attr in ("bedrock_agent_runtime_client", "bedrock_agent_client", "_boto_config"):
            self.__dict__.pop(attr, None)
        logger.debug("Authentication state reset; credentials will be refreshed on next use")
