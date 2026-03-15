"""
Main RAG pipeline orchestrating chunking, retrieval, reranking, and generation
"""
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import json

from config.settings import settings
from src.core.chunking import SemanticChunker, FixedSizeChunker, HybridChunker
from src.core.vector_store import get_vector_store
from src.core.reranker import get_reranker
from src.core.guardrails import guardrails

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Production-grade RAG pipeline with:
    - Multiple chunking strategies
    - Hybrid search
    - Cross-encoder reranking
    - Guardrails
    - Streaming support
    """
    
    def __init__(
        self,
        chunking_strategy: str = "semantic",
        reranker_type: str = "cross_encoder"
    ):
        """
        Initialize RAG pipeline
        
        Args:
            chunking_strategy: 'semantic', 'fixed', or 'hybrid'
            reranker_type: 'cross_encoder' or 'hybrid'
        """
        # Initialize components
        self.vector_store = get_vector_store()
        self.reranker = get_reranker(reranker_type)
        
        # Initialize chunker
        if chunking_strategy == "semantic":
            self.chunker = SemanticChunker()
        elif chunking_strategy == "fixed":
            self.chunker = FixedSizeChunker()
        elif chunking_strategy == "hybrid":
            self.chunker = HybridChunker()
        else:
            raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.1,
            streaming=True
        )
        
        self.chunking_strategy = chunking_strategy
        self.reranker_type = reranker_type
        
        logger.info(
            f"RAG Pipeline initialized with chunking={chunking_strategy}, "
            f"reranker={reranker_type}"
        )
    
    async def ingest_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Ingest documents into the system
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        
        Returns:
            Ingestion statistics
        """
        logger.info(f"Ingesting {len(documents)} documents...")
        
        all_chunks = []
        chunk_stats = {
            'total_documents': len(documents),
            'total_chunks': 0,
            'avg_chunk_size': 0,
            'chunking_strategy': self.chunking_strategy
        }
        
        for idx, doc in enumerate(documents):
            # Chunk document
            chunks = self.chunker.chunk(doc)
            
            # Add metadata
            doc_metadata = metadata[idx] if metadata and idx < len(metadata) else {}
            for chunk in chunks:
                chunk['metadata'] = {
                    **doc_metadata,
                    'document_id': idx,
                    'chunking_strategy': self.chunking_strategy
                }
            
            all_chunks.extend(chunks)
        
        # Add to vector store
        ids = await self.vector_store.add_documents(all_chunks)
        
        # Calculate statistics
        chunk_sizes = [len(chunk['content']) for chunk in all_chunks]
        chunk_stats['total_chunks'] = len(all_chunks)
        chunk_stats['avg_chunk_size'] = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        chunk_stats['min_chunk_size'] = min(chunk_sizes) if chunk_sizes else 0
        chunk_stats['max_chunk_size'] = max(chunk_sizes) if chunk_sizes else 0
        chunk_stats['document_ids'] = ids
        
        logger.info(f"Ingested {len(all_chunks)} chunks from {len(documents)} documents")
        
        return chunk_stats
    
    async def retrieve(
        self,
        query: str,
        use_hybrid_search: bool = True,
        top_k: int = None,
        rerank_top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            use_hybrid_search: Whether to use hybrid search
            top_k: Number of documents to retrieve initially
            rerank_top_k: Number of documents after reranking
        
        Returns:
            List of retrieved and reranked documents
        """
        top_k = top_k or settings.top_k
        rerank_top_k = rerank_top_k or settings.rerank_top_k
        
        logger.info(f"Retrieving documents for query: {query[:100]}...")
        
        # Perform search
        if use_hybrid_search:
            documents = await self.vector_store.hybrid_search(query, k=top_k)
        else:
            documents = await self.vector_store.similarity_search(query, k=top_k)
        
        logger.info(f"Retrieved {len(documents)} documents")
        
        # Rerank
        if documents and rerank_top_k < len(documents):
            reranked = self.reranker.rerank(query, documents, top_k=rerank_top_k)
            logger.info(f"Reranked to top {len(reranked)} documents")
            return reranked
        
        return documents
    
    async def generate_response(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        stream: bool = False
    ) -> str | AsyncGenerator:
        """
        Generate response using retrieved context
        
        Args:
            query: User query
            context_documents: Retrieved and reranked documents
            stream: Whether to stream the response
        
        Returns:
            Generated response (string or async generator if streaming)
        """
        # Build context from documents
        context = "\n\n".join([
            f"[Document {idx + 1}]\n{doc['content']}"
            for idx, doc in enumerate(context_documents)
        ])
        
        # Create prompt
        system_prompt = """You are a helpful AI assistant. Answer the user's question based on the provided context.
If the context doesn't contain enough information to answer the question, say so clearly.
Be concise and accurate. Cite the document number when using information from the context."""

        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        if stream:
            return self._stream_response(messages)
        else:
            response = await self.llm.ainvoke(messages)
            return response.content
    
    async def _stream_response(self, messages) -> AsyncGenerator:
        """Stream response from LLM"""
        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content
    
    async def query(
        self,
        query: str,
        use_hybrid_search: bool = True,
        stream: bool = False,
        return_sources: bool = True
    ) -> Dict[str, Any] | AsyncGenerator:
        """
        Complete RAG pipeline: retrieve and generate
        
        Args:
            query: User query
            use_hybrid_search: Use hybrid search
            stream: Stream the response
            return_sources: Include source documents in response
        
        Returns:
            Response dictionary or async generator if streaming
        """
        # Check input safety
        input_check = guardrails.check_input(query)
        if not input_check['is_safe']:
            error_message = guardrails.get_safe_response_message(input_check)
            if stream:
                async def error_stream():
                    yield error_message
                return error_stream()
            else:
                return {
                    'answer': error_message,
                    'sources': [],
                    'safety_check': input_check
                }
        
        # Retrieve documents
        documents = await self.retrieve(
            query,
            use_hybrid_search=use_hybrid_search
        )
        
        # Generate response
        if stream:
            return self._stream_query_response(query, documents, input_check)
        else:
            answer = await self.generate_response(query, documents, stream=False)
            
            # Check output safety
            output_check = guardrails.check_output(answer)
            if not output_check['is_safe']:
                answer = guardrails.get_safe_response_message(output_check)
            
            result = {
                'answer': answer,
                'query': query,
                'num_sources': len(documents),
                'input_safety_check': input_check,
                'output_safety_check': output_check
            }
            
            if return_sources:
                result['sources'] = [
                    {
                        'content': doc['content'][:200] + '...',
                        'score': doc.get('rerank_score', doc.get('score', 0)),
                        'metadata': doc.get('metadata', {})
                    }
                    for doc in documents
                ]
            
            return result
    
    async def _stream_query_response(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        input_check: Dict[str, Any]
    ) -> AsyncGenerator:
        """Stream response with safety checks"""
        # Collect full response for safety check
        full_response = ""
        
        async for chunk in self.generate_response(query, documents, stream=True):
            full_response += chunk
            yield chunk
        
        # Check output safety after streaming
        output_check = guardrails.check_output(full_response)
        if not output_check['is_safe']:
            yield f"\n\n[Response filtered due to safety concerns]"
    
    async def clear_all_documents(self):
        """Clear all documents from vector store"""
        await self.vector_store.delete_all()
        logger.info("Cleared all documents from vector store")


# Factory function
def create_rag_pipeline(
    chunking_strategy: str = "semantic",
    reranker_type: str = "cross_encoder"
) -> RAGPipeline:
    """
    Create RAG pipeline with specified configuration
    
    Args:
        chunking_strategy: 'semantic', 'fixed', or 'hybrid'
        reranker_type: 'cross_encoder' or 'hybrid'
    
    Returns:
        Configured RAG pipeline
    """
    return RAGPipeline(
        chunking_strategy=chunking_strategy,
        reranker_type=reranker_type
    )
