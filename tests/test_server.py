"""Tests for the Bedrock Knowledge Base MCP Server.

Validates tool registration, request validation, error handling,
and response formatting without requiring live AWS credentials.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.auth import BedrockAuth
from src.bedrock_client import (
    BedrockKnowledgeBaseClient,
    BedrockKnowledgeBaseError,
    KnowledgeBaseNotFoundError,
)
from src.config import RetrievalConfig, ServerConfig
from src.models import (
    KnowledgeBaseInfo,
    ListKnowledgeBasesResponse,
    MetadataFilter,
    QueryRequest,
    QueryResponse,
    RetrievalFilter,
    RetrievedPassage,
)


@pytest.fixture
def server_config() -> ServerConfig:
    """Create a test server configuration."""
    return ServerConfig(
        aws_region="us-west-2",
        aws_profile=None,
        knowledge_base_ids=["KB_TEST_001", "KB_TEST_002"],
        model_id="arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-sonnet-4-20250514",
        server_name="test-server",
        log_level="DEBUG",
        retrieval=RetrievalConfig(
            number_of_results=5,
            search_type="HYBRID",
        ),
    )


@pytest.fixture
def mock_auth(server_config: ServerConfig) -> MagicMock:
    """Create a mock BedrockAuth instance."""
    auth = MagicMock(spec=BedrockAuth)
    auth.bedrock_agent_runtime_client = MagicMock()
    auth.bedrock_agent_client = MagicMock()
    return auth


@pytest.fixture
def client(server_config: ServerConfig, mock_auth: MagicMock) -> BedrockKnowledgeBaseClient:
    """Create a BedrockKnowledgeBaseClient with mocked auth."""
    return BedrockKnowledgeBaseClient(server_config, mock_auth)


class TestServerConfig:
    """Tests for configuration loading and validation."""

    def test_default_config(self) -> None:
        """Default configuration uses expected values."""
        config = ServerConfig()
        assert config.aws_region == "us-west-2"
        assert config.knowledge_base_ids == []
        assert config.log_level == "INFO"
        assert config.retrieval.number_of_results == 5
        assert config.retrieval.search_type == "HYBRID"

    def test_config_with_kb_ids(self) -> None:
        """Configuration properly stores knowledge base IDs."""
        config = ServerConfig(knowledge_base_ids=["KB001", "KB002"])
        assert config.knowledge_base_ids == ["KB001", "KB002"]

    def test_retrieval_config_bounds(self) -> None:
        """Retrieval config enforces valid bounds."""
        config = RetrievalConfig(number_of_results=50)
        assert config.number_of_results == 50

        with pytest.raises(Exception):
            RetrievalConfig(number_of_results=0)

        with pytest.raises(Exception):
            RetrievalConfig(number_of_results=101)


class TestModels:
    """Tests for Pydantic request/response models."""

    def test_query_request_validation(self) -> None:
        """QueryRequest validates required fields and constraints."""
        req = QueryRequest(
            knowledge_base_id="KB001",
            query="test query",
        )
        assert req.generate_response is True
        assert req.number_of_results is None

    def test_query_request_empty_query_rejected(self) -> None:
        """QueryRequest rejects empty query strings."""
        with pytest.raises(Exception):
            QueryRequest(knowledge_base_id="KB001", query="")

    def test_metadata_filter_model(self) -> None:
        """MetadataFilter accepts valid filter configurations."""
        f = MetadataFilter(key="department", value="claims", operator="equals")
        assert f.key == "department"
        assert f.value == "claims"
        assert f.operator == "equals"

    def test_retrieval_filter_and_conditions(self) -> None:
        """RetrievalFilter properly stores AND conditions."""
        rf = RetrievalFilter(
            and_conditions=[
                MetadataFilter(key="type", value="policy", operator="equals"),
                MetadataFilter(key="year", value=2024, operator="greater_than"),
            ]
        )
        assert len(rf.and_conditions) == 2
        assert rf.or_conditions is None

    def test_query_response_serialization(self) -> None:
        """QueryResponse serializes to expected format."""
        resp = QueryResponse(
            answer="Test answer",
            passages=[
                RetrievedPassage(
                    content="Passage text",
                    score=0.95,
                    source_uri="s3://bucket/doc.pdf",
                    source_metadata={"page": 5},
                )
            ],
            knowledge_base_id="KB001",
            query="test",
            citation_count=1,
        )
        data = resp.model_dump()
        assert data["answer"] == "Test answer"
        assert len(data["passages"]) == 1
        assert data["passages"][0]["score"] == 0.95


class TestBedrockClient:
    """Tests for the BedrockKnowledgeBaseClient."""

    @pytest.mark.asyncio
    async def test_query_validates_kb_id(
        self, client: BedrockKnowledgeBaseClient
    ) -> None:
        """Query rejects knowledge base IDs not in the allowed list."""
        request = QueryRequest(
            knowledge_base_id="KB_NOT_ALLOWED",
            query="test query",
        )
        with pytest.raises(BedrockKnowledgeBaseError, match="not in the configured"):
            await client.query(request)

    @pytest.mark.asyncio
    async def test_query_allowed_kb_id(
        self, client: BedrockKnowledgeBaseClient, mock_auth: MagicMock
    ) -> None:
        """Query proceeds for allowed knowledge base IDs."""
        mock_auth.bedrock_agent_runtime_client.retrieve_and_generate.return_value = {
            "output": {"text": "Generated answer"},
            "citations": [
                {
                    "retrievedReferences": [
                        {
                            "content": {"text": "Source passage"},
                            "location": {"s3Location": {"uri": "s3://bucket/doc.pdf"}},
                            "metadata": {"page": "3"},
                        }
                    ]
                }
            ],
        }

        request = QueryRequest(
            knowledge_base_id="KB_TEST_001",
            query="What is the claims process?",
            generate_response=True,
        )
        response = await client.query(request)

        assert response.answer == "Generated answer"
        assert len(response.passages) == 1
        assert response.passages[0].source_uri == "s3://bucket/doc.pdf"
        assert response.citation_count == 1

    @pytest.mark.asyncio
    async def test_retrieve_without_generation(
        self, client: BedrockKnowledgeBaseClient, mock_auth: MagicMock
    ) -> None:
        """Retrieve returns raw passages without model generation."""
        mock_auth.bedrock_agent_runtime_client.retrieve.return_value = {
            "retrievalResults": [
                {
                    "content": {"text": "First passage"},
                    "score": 0.92,
                    "location": {"s3Location": {"uri": "s3://bucket/a.pdf"}},
                    "metadata": {},
                },
                {
                    "content": {"text": "Second passage"},
                    "score": 0.85,
                    "location": {"s3Location": {"uri": "s3://bucket/b.pdf"}},
                    "metadata": {"category": "claims"},
                },
            ]
        }

        request = QueryRequest(
            knowledge_base_id="KB_TEST_001",
            query="claims handling procedure",
            generate_response=False,
            number_of_results=3,
        )
        response = await client.query(request)

        assert response.answer is None
        assert len(response.passages) == 2
        assert response.passages[0].score == 0.92
        assert response.passages[1].source_metadata == {"category": "claims"}

    @pytest.mark.asyncio
    async def test_list_knowledge_bases(
        self, client: BedrockKnowledgeBaseClient, mock_auth: MagicMock
    ) -> None:
        """List knowledge bases returns filtered results."""
        mock_auth.bedrock_agent_client.list_knowledge_bases.return_value = {
            "knowledgeBaseSummaries": [
                {
                    "knowledgeBaseId": "KB_TEST_001",
                    "name": "Insurance Policies",
                    "description": "Policy documents and guidelines",
                    "status": "ACTIVE",
                },
                {
                    "knowledgeBaseId": "KB_OTHER",
                    "name": "Other KB",
                    "status": "ACTIVE",
                },
            ]
        }

        response = await client.list_knowledge_bases(max_results=10)

        # Should filter to only configured KB IDs
        assert len(response.knowledge_bases) == 1
        assert response.knowledge_bases[0].knowledge_base_id == "KB_TEST_001"
        assert response.knowledge_bases[0].name == "Insurance Policies"

    @pytest.mark.asyncio
    async def test_build_filter_and_conditions(
        self, client: BedrockKnowledgeBaseClient
    ) -> None:
        """Filter builder correctly formats AND conditions."""
        retrieval_filter = RetrievalFilter(
            and_conditions=[
                MetadataFilter(key="type", value="policy", operator="equals"),
                MetadataFilter(key="year", value=2024, operator="greater_than"),
            ]
        )
        result = client._build_filter(retrieval_filter)

        assert "andAll" in result
        assert len(result["andAll"]) == 2
        assert result["andAll"][0] == {"equals": {"key": "type", "value": "policy"}}
        assert result["andAll"][1] == {"greaterThan": {"key": "year", "value": 2024}}

    @pytest.mark.asyncio
    async def test_build_filter_or_conditions(
        self, client: BedrockKnowledgeBaseClient
    ) -> None:
        """Filter builder correctly formats OR conditions."""
        retrieval_filter = RetrievalFilter(
            or_conditions=[
                MetadataFilter(key="state", value="FL", operator="equals"),
                MetadataFilter(key="state", value="TX", operator="equals"),
            ]
        )
        result = client._build_filter(retrieval_filter)

        assert "orAll" in result
        assert len(result["orAll"]) == 2


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_resource_not_found_error(
        self, client: BedrockKnowledgeBaseClient, mock_auth: MagicMock
    ) -> None:
        """ResourceNotFoundException maps to KnowledgeBaseNotFoundError."""
        from botocore.exceptions import ClientError

        mock_auth.bedrock_agent_runtime_client.retrieve_and_generate.side_effect = (
            ClientError(
                {"Error": {"Code": "ResourceNotFoundException", "Message": "KB not found"}},
                "RetrieveAndGenerate",
            )
        )

        request = QueryRequest(
            knowledge_base_id="KB_TEST_001",
            query="test",
            generate_response=True,
        )

        with pytest.raises(KnowledgeBaseNotFoundError):
            await client.query(request)

    @pytest.mark.asyncio
    async def test_throttling_error(
        self, client: BedrockKnowledgeBaseClient, mock_auth: MagicMock
    ) -> None:
        """ThrottlingException is properly wrapped."""
        from botocore.exceptions import ClientError

        mock_auth.bedrock_agent_runtime_client.retrieve.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "Retrieve",
        )

        request = QueryRequest(
            knowledge_base_id="KB_TEST_001",
            query="test",
            generate_response=False,
        )

        with pytest.raises(BedrockKnowledgeBaseError, match="Rate limited"):
            await client.query(request)
