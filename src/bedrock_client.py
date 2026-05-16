"""Wrapper around Amazon Bedrock Knowledge Base APIs.

Provides a high-level interface for Retrieve, RetrieveAndGenerate, and
management operations against Bedrock Knowledge Bases, with proper error
handling, retry logic, and response normalization.
"""

from __future__ import annotations

import logging
from typing import Any

from botocore.exceptions import ClientError

from .auth import BedrockAuth
from .config import ServerConfig
from .models import (
    DocumentMetadata,
    GetDocumentMetadataResponse,
    KnowledgeBaseInfo,
    ListKnowledgeBasesResponse,
    MetadataFilter,
    QueryRequest,
    QueryResponse,
    RetrievalFilter,
    RetrievedPassage,
)

logger = logging.getLogger(__name__)


class BedrockKnowledgeBaseError(Exception):
    """Base exception for Bedrock Knowledge Base operations."""

    def __init__(self, message: str, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


class KnowledgeBaseNotFoundError(BedrockKnowledgeBaseError):
    """Raised when a referenced knowledge base does not exist."""


class BedrockKnowledgeBaseClient:
    """Client for Amazon Bedrock Knowledge Base operations.

    Wraps the Bedrock Agent Runtime and Bedrock Agent APIs, providing
    typed request/response models and consistent error handling.

    Args:
        config: Server configuration.
        auth: Authentication manager providing boto3 clients.
    """

    def __init__(self, config: ServerConfig, auth: BedrockAuth) -> None:
        self._config = config
        self._auth = auth

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Query a knowledge base with optional metadata filtering.

        Routes to RetrieveAndGenerate (for synthesized answers) or Retrieve
        (for raw passages) based on the request's generate_response flag.

        Args:
            request: Validated query parameters including KB ID, query text,
                     filters, and retrieval configuration.

        Returns:
            QueryResponse with answer and/or retrieved passages.

        Raises:
            KnowledgeBaseNotFoundError: If the knowledge base ID is invalid.
            BedrockKnowledgeBaseError: For other API failures.
        """
        self._validate_knowledge_base_id(request.knowledge_base_id)

        if request.generate_response:
            return await self._retrieve_and_generate(request)
        else:
            return await self._retrieve(request)

    async def _retrieve_and_generate(self, request: QueryRequest) -> QueryResponse:
        """Execute a RetrieveAndGenerate call for a synthesized answer.

        Uses the configured foundation model to generate a response
        grounded in the retrieved knowledge base passages.
        """
        try:
            retrieval_config = self._build_retrieval_config(request)

            api_params: dict[str, Any] = {
                "input": {"text": request.query},
                "retrieveAndGenerateConfiguration": {
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": request.knowledge_base_id,
                        "modelArn": self._config.model_id,
                        "retrievalConfiguration": {
                            "vectorSearchConfiguration": retrieval_config
                        },
                    },
                },
            }

            logger.debug(
                "RetrieveAndGenerate request for KB %s: query=%r",
                request.knowledge_base_id,
                request.query[:100],
            )

            response = self._auth.bedrock_agent_runtime_client.retrieve_and_generate(
                **api_params
            )

            # Extract answer
            answer = response.get("output", {}).get("text", "")

            # Extract citations and passages
            passages: list[RetrievedPassage] = []
            citations = response.get("citations", [])
            for citation in citations:
                for ref in citation.get("retrievedReferences", []):
                    content = ref.get("content", {}).get("text", "")
                    location = ref.get("location", {})
                    source_uri = location.get("s3Location", {}).get("uri", "")
                    metadata = ref.get("metadata", {})

                    passages.append(
                        RetrievedPassage(
                            content=content,
                            source_uri=source_uri,
                            source_metadata=metadata,
                        )
                    )

            return QueryResponse(
                answer=answer,
                passages=passages,
                knowledge_base_id=request.knowledge_base_id,
                query=request.query,
                citation_count=len(citations),
            )

        except ClientError as e:
            self._handle_client_error(e, request.knowledge_base_id)
            raise  # unreachable, _handle_client_error always raises

    async def _retrieve(self, request: QueryRequest) -> QueryResponse:
        """Execute a Retrieve call for raw passages without generation.

        Returns scored passages directly from the vector search without
        passing them through a foundation model.
        """
        try:
            retrieval_config = self._build_retrieval_config(request)
            number_of_results = (
                request.number_of_results or self._config.retrieval.number_of_results
            )

            api_params: dict[str, Any] = {
                "knowledgeBaseId": request.knowledge_base_id,
                "retrievalQuery": {"text": request.query},
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        **retrieval_config,
                        "numberOfResults": number_of_results,
                    }
                },
            }

            logger.debug(
                "Retrieve request for KB %s: query=%r, num_results=%d",
                request.knowledge_base_id,
                request.query[:100],
                number_of_results,
            )

            response = self._auth.bedrock_agent_runtime_client.retrieve(**api_params)

            passages: list[RetrievedPassage] = []
            for result in response.get("retrievalResults", []):
                content = result.get("content", {}).get("text", "")
                score = result.get("score")
                location = result.get("location", {})
                source_uri = location.get("s3Location", {}).get("uri", "")
                metadata = result.get("metadata", {})

                passages.append(
                    RetrievedPassage(
                        content=content,
                        score=score,
                        source_uri=source_uri,
                        source_metadata=metadata,
                    )
                )

            return QueryResponse(
                answer=None,
                passages=passages,
                knowledge_base_id=request.knowledge_base_id,
                query=request.query,
                citation_count=0,
            )

        except ClientError as e:
            self._handle_client_error(e, request.knowledge_base_id)
            raise

    async def list_knowledge_bases(
        self, max_results: int = 10, next_token: str | None = None
    ) -> ListKnowledgeBasesResponse:
        """List available knowledge bases with pagination support.

        If the server is configured with specific KB IDs, only those are returned.
        Otherwise, lists all KBs accessible to the authenticated principal.

        Args:
            max_results: Maximum number of results per page.
            next_token: Pagination token from previous response.

        Returns:
            ListKnowledgeBasesResponse with KB summaries and pagination token.
        """
        try:
            api_params: dict[str, Any] = {"maxResults": max_results}
            if next_token:
                api_params["nextToken"] = next_token

            response = self._auth.bedrock_agent_client.list_knowledge_bases(**api_params)

            knowledge_bases: list[KnowledgeBaseInfo] = []
            for kb_summary in response.get("knowledgeBaseSummaries", []):
                kb_id = kb_summary.get("knowledgeBaseId", "")

                # If configured with specific IDs, filter to only those
                if self._config.knowledge_base_ids and kb_id not in self._config.knowledge_base_ids:
                    continue

                knowledge_bases.append(
                    KnowledgeBaseInfo(
                        knowledge_base_id=kb_id,
                        name=kb_summary.get("name", ""),
                        description=kb_summary.get("description"),
                        status=kb_summary.get("status", "UNKNOWN"),
                        updated_at=kb_summary.get("updatedAt"),
                    )
                )

            return ListKnowledgeBasesResponse(
                knowledge_bases=knowledge_bases,
                next_token=response.get("nextToken"),
            )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            raise BedrockKnowledgeBaseError(
                f"Failed to list knowledge bases: {e.response['Error']['Message']}",
                error_code=error_code,
            ) from e

    async def get_document_metadata(
        self,
        knowledge_base_id: str,
        data_source_id: str,
        document_uri: str | None = None,
    ) -> GetDocumentMetadataResponse:
        """Retrieve metadata for documents in a knowledge base data source.

        Lists documents ingested into the specified data source, optionally
        filtered by document URI.

        Args:
            knowledge_base_id: Target knowledge base ID.
            data_source_id: Data source within the knowledge base.
            document_uri: Optional URI filter for a specific document.

        Returns:
            GetDocumentMetadataResponse with document metadata entries.
        """
        self._validate_knowledge_base_id(knowledge_base_id)

        try:
            api_params: dict[str, Any] = {
                "knowledgeBaseId": knowledge_base_id,
                "dataSourceId": data_source_id,
            }

            if document_uri:
                api_params["filters"] = [
                    {
                        "attribute": "URI",
                        "operator": "EQUALS",
                        "values": [document_uri],
                    }
                ]

            response = self._auth.bedrock_agent_client.list_data_source_sync_jobs(
                **api_params
            )

            # Normalize the response into our document metadata model
            documents: list[DocumentMetadata] = []
            for item in response.get("dataSourceSyncJobs", []):
                documents.append(
                    DocumentMetadata(
                        document_uri=item.get("dataSourceId", data_source_id),
                        status=item.get("status", "UNKNOWN"),
                        metadata=item.get("metrics", {}),
                        last_updated=item.get("lastUpdatedAt"),
                    )
                )

            return GetDocumentMetadataResponse(
                documents=documents,
                knowledge_base_id=knowledge_base_id,
                data_source_id=data_source_id,
            )

        except ClientError as e:
            self._handle_client_error(e, knowledge_base_id)
            raise

    def _build_retrieval_config(self, request: QueryRequest) -> dict[str, Any]:
        """Build the vectorSearchConfiguration for a retrieval request.

        Applies metadata filters and search type overrides based on
        request parameters and server defaults.
        """
        config: dict[str, Any] = {}

        # Apply search type
        search_type = request.search_type or self._config.retrieval.search_type
        if self._config.retrieval.override_search_type or request.search_type:
            config["overrideSearchType"] = search_type

        # Apply metadata filters
        if request.filter:
            config["filter"] = self._build_filter(request.filter)

        return config

    def _build_filter(self, retrieval_filter: RetrievalFilter) -> dict[str, Any]:
        """Convert our filter model to Bedrock API filter format.

        Translates MetadataFilter conditions into the nested filter structure
        expected by the Bedrock Retrieve API.
        """
        if retrieval_filter.and_conditions:
            return {
                "andAll": [
                    self._build_single_filter(f) for f in retrieval_filter.and_conditions
                ]
            }
        elif retrieval_filter.or_conditions:
            return {
                "orAll": [
                    self._build_single_filter(f) for f in retrieval_filter.or_conditions
                ]
            }
        return {}

    def _build_single_filter(self, metadata_filter: MetadataFilter) -> dict[str, Any]:
        """Convert a single MetadataFilter to Bedrock API format."""
        operator_map = {
            "equals": "equals",
            "not_equals": "notEquals",
            "greater_than": "greaterThan",
            "less_than": "lessThan",
            "greater_than_or_equals": "greaterThanOrEquals",
            "less_than_or_equals": "lessThanOrEquals",
            "in": "in",
            "not_in": "notIn",
            "starts_with": "startsWith",
            "contains": "listContains",
        }

        api_operator = operator_map.get(metadata_filter.operator, metadata_filter.operator)
        return {
            api_operator: {
                "key": metadata_filter.key,
                "value": metadata_filter.value,
            }
        }

    def _validate_knowledge_base_id(self, knowledge_base_id: str) -> None:
        """Validate that a KB ID is in the allowed list (if configured)."""
        if (
            self._config.knowledge_base_ids
            and knowledge_base_id not in self._config.knowledge_base_ids
        ):
            raise BedrockKnowledgeBaseError(
                f"Knowledge base '{knowledge_base_id}' is not in the configured "
                f"allowed list: {self._config.knowledge_base_ids}",
                error_code="VALIDATION_ERROR",
            )

    def _handle_client_error(self, error: ClientError, knowledge_base_id: str) -> None:
        """Map AWS ClientError to domain-specific exceptions."""
        error_code = error.response["Error"]["Code"]
        message = error.response["Error"]["Message"]

        if error_code == "ResourceNotFoundException":
            raise KnowledgeBaseNotFoundError(
                f"Knowledge base '{knowledge_base_id}' not found: {message}",
                error_code=error_code,
            ) from error
        elif error_code == "ValidationException":
            raise BedrockKnowledgeBaseError(
                f"Invalid request for KB '{knowledge_base_id}': {message}",
                error_code=error_code,
            ) from error
        elif error_code == "ThrottlingException":
            raise BedrockKnowledgeBaseError(
                f"Rate limited querying KB '{knowledge_base_id}': {message}. "
                "Consider reducing request frequency.",
                error_code=error_code,
            ) from error
        elif error_code in ("AccessDeniedException", "UnauthorizedAccessException"):
            raise BedrockKnowledgeBaseError(
                f"Access denied for KB '{knowledge_base_id}': {message}. "
                "Check IAM permissions.",
                error_code=error_code,
            ) from error
        else:
            raise BedrockKnowledgeBaseError(
                f"Bedrock API error ({error_code}) for KB '{knowledge_base_id}': {message}",
                error_code=error_code,
            ) from error
