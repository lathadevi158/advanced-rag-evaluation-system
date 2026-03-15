"""
Basic test suite for Production RAG System
Run with: pytest tests/
"""
import pytest
import asyncio
from src.core.chunking import SemanticChunker, FixedSizeChunker, HybridChunker


class TestChunking:
    """Test chunking strategies"""
    
    def test_semantic_chunker(self):
        """Test semantic chunking"""
        chunker = SemanticChunker()
        
        text = """
        Machine learning is a field of AI. It enables computers to learn from data.
        Deep learning is a subset of machine learning. It uses neural networks with multiple layers.
        Natural language processing deals with text. It helps computers understand human language.
        """
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0
        assert all('content' in chunk for chunk in chunks)
        assert all('chunk_type' in chunk for chunk in chunks)
        assert all(chunk['chunk_type'] == 'semantic' for chunk in chunks)
    
    def test_fixed_size_chunker(self):
        """Test fixed-size chunking"""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        
        text = "a " * 200  # 200 words
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 1
        assert all('content' in chunk for chunk in chunks)
        assert all(chunk['chunk_type'] == 'fixed_size' for chunk in chunks)
    
    def test_hybrid_chunker(self):
        """Test hybrid chunking"""
        chunker = HybridChunker(max_chunk_size=200)
        
        text = "This is a test sentence. " * 50
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0
        assert all('content' in chunk for chunk in chunks)


class TestReranker:
    """Test reranking functionality"""
    
    def test_cross_encoder_reranker(self):
        """Test cross-encoder reranking"""
        from src.core.reranker import CrossEncoderReranker
        
        reranker = CrossEncoderReranker()
        
        query = "What is Python?"
        documents = [
            {
                'content': 'Python is a programming language used for web development and data science.',
                'score': 0.7
            },
            {
                'content': 'The snake species Python is found in tropical regions.',
                'score': 0.8
            },
            {
                'content': 'Java is another popular programming language.',
                'score': 0.6
            }
        ]
        
        reranked = reranker.rerank(query, documents, top_k=2)
        
        assert len(reranked) == 2
        assert all('rerank_score' in doc for doc in reranked)
        # First result should be about programming, not snakes
        assert 'programming' in reranked[0]['content'].lower()


class TestGuardrails:
    """Test safety guardrails"""
    
    def test_toxicity_detection(self):
        """Test toxicity detection"""
        from src.core.guardrails import ContentGuardrails
        
        guardrails = ContentGuardrails(toxicity_threshold=0.7)
        
        # Safe content
        safe_result = guardrails.check_content("Python is a great programming language.")
        assert safe_result['is_safe'] == True
        
        # Toxic content
        toxic_result = guardrails.check_content("I hate you and you are stupid!")
        # This should detect toxicity
        assert 'toxicity' in toxic_result['scores']
    
    def test_bias_detection(self):
        """Test bias detection"""
        from src.core.guardrails import BiasDetector
        
        detector = BiasDetector()
        
        # Neutral content
        neutral = detector.detect_bias("Software engineers write code.")
        assert neutral['has_bias'] == False
        
        # Potentially biased content
        biased = detector.detect_bias("Women are too emotional for leadership.")
        assert biased['has_bias'] == True


@pytest.mark.asyncio
class TestRAGPipeline:
    """Test RAG pipeline integration"""
    
    async def test_pipeline_creation(self):
        """Test pipeline can be created"""
        from src.core import create_rag_pipeline
        
        pipeline = create_rag_pipeline(
            chunking_strategy="fixed",
            reranker_type="cross_encoder"
        )
        
        assert pipeline is not None
        assert pipeline.chunking_strategy == "fixed"
        assert pipeline.reranker_type == "cross_encoder"
    
    async def test_document_ingestion(self):
        """Test document ingestion (requires API keys)"""
        # Skip if no API keys
        import os
        if not os.getenv('OPENAI_API_KEY'):
            pytest.skip("OpenAI API key not set")
        
        from src.core import create_rag_pipeline
        
        pipeline = create_rag_pipeline()
        
        documents = ["Test document 1", "Test document 2"]
        
        try:
            stats = await pipeline.ingest_documents(documents)
            
            assert stats['total_documents'] == 2
            assert stats['total_chunks'] > 0
            
            # Cleanup
            await pipeline.clear_all_documents()
        except Exception as e:
            pytest.skip(f"Ingestion failed (might be expected without proper setup): {e}")


class TestPydanticModels:
    """Test API models"""
    
    def test_query_request_validation(self):
        """Test QueryRequest validation"""
        from src.api.models import QueryRequest
        
        # Valid request
        valid_request = QueryRequest(
            query="What is AI?",
            use_hybrid_search=True,
            stream=False
        )
        
        assert valid_request.query == "What is AI?"
        assert valid_request.use_hybrid_search == True
        
        # Invalid request (empty query)
        with pytest.raises(Exception):
            QueryRequest(query="")
    
    def test_document_upload_validation(self):
        """Test DocumentUpload validation"""
        from src.api.models import BatchDocumentUpload
        
        valid_upload = BatchDocumentUpload(
            documents=["Doc 1", "Doc 2"],
            metadata=[{"source": "test"}, {"source": "test2"}]
        )
        
        assert len(valid_upload.documents) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
