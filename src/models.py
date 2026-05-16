"""Pydantic models for request/response validation.

Defines strongly-typed models for Bedrock Knowledge Base API interactions,
ensuring data integrity at the boundary between the MCP server and AWS APIs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# --- Request Models ---


class MetadataFilter(BaseModel):
    """A single metadata filter condition for knowledge base retrieval.

    Supports equality, comparison, and membership operators against
    document metadata attributes stored in the knowledge base.
    """

    key: str = Field(description="Metadata attribute key to filter on")
    value: str | int | float | bool | list[str] = Field(
        description="Value to compare against"
    )
    operator: str = Field(
        default="equals",
        description="Filter operator: equals, not_equals, greater_than, less_than, in, not_in",
    )


class RetrievalFilter(BaseModel):
    """Composite filter supporting AND/OR logic for metadata filtering."""

    and_conditions: list[MetadataFilter] | None = Field(
        default=None, description="All conditions must match (AND logic)"
    )
    or_conditions: list[MetadataFilter] | None = Field(
        default=None, description="Any condition must match (OR logic)"
    )


class QueryRequest(BaseModel):
    """Request model for querying a knowledge base."""

    knowledge_base_id: str = Field(description="Target knowledge base ID")
    query: str = Field(description="Natural language query text", min_length=1, max_length=1000)
    number_of_results: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of results to retrieve (overrides server default)",
    )
    filter: RetrievalFilter | None = Field(
        default=None, description="Metadata filter to narrow results"
    )
    search_type: str | None = Field(
        default=None,
        description="Override search type: HYBRID or SEMANTIC",
    )
    generate_response: bool = Field(
        default=True,
        description="If True, uses RetrieveAndGenerate for a synthesized answer. "
        "If False, returns raw retrieved passages.",
    )


class ListKnowledgeBasesRequest(BaseModel):
    """Request model for listing available knowledge bases."""

    max_results: int = Field(default=10, ge=1, le=100)
    next_token: str | None = Field(default=None, description="Pagination token")


class GetDocumentMetadataRequest(BaseModel):
    """Request model for retrieving document metadata from a knowledge base."""

    knowledge_base_id: str = Field(description="Knowledge base ID")
    data_source_id: str = Field(description="Data source ID within the knowledge base")
    document_uri: str | None = Field(
        default=None, description="Filter by specific document URI"
    )


# --- Response Models ---


class RetrievedPassage(BaseModel):
    """A single passage retrieved from the knowledge base."""

    content: str = Field(description="Text content of the retrieved passage")
    score: float | None = Field(default=None, description="Relevance score (0-1)")
    source_uri: str | None = Field(default=None, description="Source document URI")
    source_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata attributes from the source document"
    )


class QueryResponse(BaseModel):
    """Response from a knowledge base query."""

    answer: str | None = Field(
        default=None, description="Generated answer (only with generate_response=True)"
    )
    passages: list[RetrievedPassage] = Field(
        default_factory=list, description="Retrieved passages from the knowledge base"
    )
    knowledge_base_id: str = Field(description="Knowledge base that was queried")
    query: str = Field(description="Original query text")
    citation_count: int = Field(default=0, description="Number of citations in the answer")


class KnowledgeBaseInfo(BaseModel):
    """Summary information about a knowledge base."""

    knowledge_base_id: str = Field(description="Unique knowledge base identifier")
    name: str = Field(description="Human-readable name")
    description: str | None = Field(default=None, description="Knowledge base description")
    status: str = Field(description="Current status (ACTIVE, CREATING, DELETING, etc.)")
    updated_at: datetime | None = Field(default=None, description="Last update timestamp")


class ListKnowledgeBasesResponse(BaseModel):
    """Response from listing knowledge bases."""

    knowledge_bases: list[KnowledgeBaseInfo] = Field(default_factory=list)
    next_token: str | None = Field(default=None, description="Pagination token for next page")


class DocumentMetadata(BaseModel):
    """Metadata about a document in a knowledge base data source."""

    document_uri: str = Field(description="S3 URI or web URL of the document")
    status: str = Field(description="Ingestion status")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata attributes"
    )
    last_updated: datetime | None = Field(default=None)


class GetDocumentMetadataResponse(BaseModel):
    """Response from getting document metadata."""

    documents: list[DocumentMetadata] = Field(default_factory=list)
    knowledge_base_id: str
    data_source_id: str
