"""
Configuration management using Pydantic Settings
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(None, env="ANTHROPIC_API_KEY")
    
    # Vector Database Settings
    vector_db: Literal["pinecone", "weaviate"] = Field("pinecone", env="VECTOR_DB")
    
    # Pinecone
    pinecone_api_key: str | None = Field(None, env="PINECONE_API_KEY")
    pinecone_environment: str = Field("us-east-1-aws", env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field("production-rag", env="PINECONE_INDEX_NAME")
    
    # Weaviate
    weaviate_url: str = Field("http://localhost:8080", env="WEAVIATE_URL")
    weaviate_api_key: str | None = Field(None, env="WEAVIATE_API_KEY")
    
    # LangSmith
    langchain_tracing_v2: bool = Field(False, env="LANGCHAIN_TRACING_V2")
    langchain_api_key: str | None = Field(None, env="LANGCHAIN_API_KEY")
    langchain_project: str = Field("production-rag-system", env="LANGCHAIN_PROJECT")
    
    # Model Settings
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    llm_model: str = Field("gpt-4-turbo-preview", env="LLM_MODEL")
    
    # RAG Settings
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP")
    top_k: int = Field(10, env="TOP_K")
    rerank_top_k: int = Field(3, env="RERANK_TOP_K")
    
    # Guardrails
    enable_guardrails: bool = Field(True, env="ENABLE_GUARDRAILS")
    toxicity_threshold: float = Field(0.7, env="TOXICITY_THRESHOLD")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
