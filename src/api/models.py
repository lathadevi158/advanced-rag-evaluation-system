"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class DocumentUpload(BaseModel):
    """Request model for document upload"""
    content: str = Field(..., description="Document text content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class BatchDocumentUpload(BaseModel):
    """Request model for batch document upload"""
    documents: List[str] = Field(..., description="List of document texts")
    metadata: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional metadata per document")


class QueryRequest(BaseModel):
    """Request model for RAG query"""
    query: str = Field(..., description="User question", min_length=1)
    use_hybrid_search: bool = Field(default=True, description="Use hybrid search (vector + keyword)")
    stream: bool = Field(default=False, description="Stream response")
    return_sources: bool = Field(default=True, description="Include source documents in response")
    top_k: Optional[int] = Field(default=None, description="Number of documents to retrieve", ge=1, le=50)
    rerank_top_k: Optional[int] = Field(default=None, description="Number of documents after reranking", ge=1, le=20)


class SourceDocument(BaseModel):
    """Source document in response"""
    content: str = Field(..., description="Document content (truncated)")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class QueryResponse(BaseModel):
    """Response model for RAG query"""
    answer: str = Field(..., description="Generated answer")
    query: str = Field(..., description="Original query")
    num_sources: int = Field(..., description="Number of source documents used")
    sources: Optional[List[SourceDocument]] = Field(default=None, description="Source documents")
    input_safety_check: Dict[str, Any] = Field(..., description="Input safety check results")
    output_safety_check: Dict[str, Any] = Field(..., description="Output safety check results")


class IngestionResponse(BaseModel):
    """Response model for document ingestion"""
    total_documents: int = Field(..., description="Number of documents ingested")
    total_chunks: int = Field(..., description="Number of chunks created")
    avg_chunk_size: float = Field(..., description="Average chunk size")
    chunking_strategy: str = Field(..., description="Chunking strategy used")
    message: str = Field(..., description="Success message")


class EvaluationRequest(BaseModel):
    """Request model for evaluation"""
    questions: List[str] = Field(..., description="User questions")
    answers: List[str] = Field(..., description="Generated answers")
    contexts: List[List[str]] = Field(..., description="Retrieved contexts per question")
    ground_truths: List[str] = Field(..., description="Expected ground truth answers")
    save_results: bool = Field(default=True, description="Save evaluation results to file")


class EvaluationResponse(BaseModel):
    """Response model for evaluation"""
    timestamp: str = Field(..., description="Evaluation timestamp")
    num_examples: int = Field(..., description="Number of examples evaluated")
    ragas_overall_score: float = Field(..., description="RAGAS overall score")
    deepeval_overall_score: float = Field(..., description="DeepEval overall score")
    average_score: float = Field(..., description="Average of both frameworks")
    full_report: str = Field(..., description="Detailed evaluation report")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    vector_db: str = Field(..., description="Vector database in use")
    guardrails_enabled: bool = Field(..., description="Guardrails status")
