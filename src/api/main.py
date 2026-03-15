"""
FastAPI application for Production RAG System
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List
import PyPDF2
import io

from config.settings import settings
from src.core import create_rag_pipeline
from src.evaluation import create_comprehensive_evaluator
from src.api.models import (
    QueryRequest, QueryResponse, SourceDocument,
    BatchDocumentUpload, IngestionResponse,
    EvaluationRequest, EvaluationResponse,
    HealthResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Production RAG System",
    description="Advanced RAG system with hybrid search, reranking, and comprehensive evaluation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline (global instance)
rag_pipeline = None
evaluator = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global rag_pipeline, evaluator
    
    logger.info("Starting up Production RAG System...")
    logger.info(f"Vector DB: {settings.vector_db}")
    logger.info(f"Embedding Model: {settings.embedding_model}")
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"Guardrails: {'Enabled' if settings.enable_guardrails else 'Disabled'}")
    
    try:
        # Initialize RAG pipeline
        rag_pipeline = create_rag_pipeline(
            chunking_strategy="semantic",
            reranker_type="cross_encoder"
        )
        logger.info("RAG pipeline initialized successfully")
        
        # Initialize evaluator
        evaluator = create_comprehensive_evaluator()
        logger.info("Evaluator initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Production RAG System...")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vector_db=settings.vector_db,
        guardrails_enabled=settings.enable_guardrails
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vector_db=settings.vector_db,
        guardrails_enabled=settings.enable_guardrails
    )


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(request: BatchDocumentUpload):
    """
    Ingest multiple documents into the system
    
    Documents are chunked and stored in the vector database
    """
    try:
        logger.info(f"Ingesting {len(request.documents)} documents...")
        
        stats = await rag_pipeline.ingest_documents(
            documents=request.documents,
            metadata=request.metadata
        )
        
        return IngestionResponse(
            total_documents=stats['total_documents'],
            total_chunks=stats['total_chunks'],
            avg_chunk_size=stats['avg_chunk_size'],
            chunking_strategy=stats['chunking_strategy'],
            message=f"Successfully ingested {stats['total_documents']} documents into {stats['total_chunks']} chunks"
        )
        
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Ingest PDF document
    
    Extracts text from PDF and ingests it
    """
    try:
        # Read PDF
        pdf_bytes = await file.read()
        pdf_file = io.BytesIO(pdf_bytes)
        
        # Extract text
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Ingest
        stats = await rag_pipeline.ingest_documents(
            documents=[text],
            metadata=[{"filename": file.filename, "type": "pdf"}]
        )
        
        return IngestionResponse(
            total_documents=stats['total_documents'],
            total_chunks=stats['total_chunks'],
            avg_chunk_size=stats['avg_chunk_size'],
            chunking_strategy=stats['chunking_strategy'],
            message=f"Successfully ingested PDF: {file.filename}"
        )
        
    except Exception as e:
        logger.error(f"Error ingesting PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system
    
    Retrieves relevant documents and generates an answer
    """
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Execute RAG pipeline
        result = await rag_pipeline.query(
            query=request.query,
            use_hybrid_search=request.use_hybrid_search,
            stream=False,
            return_sources=request.return_sources
        )
        
        # Convert to response model
        response = QueryResponse(
            answer=result['answer'],
            query=result['query'],
            num_sources=result['num_sources'],
            input_safety_check=result['input_safety_check'],
            output_safety_check=result['output_safety_check']
        )
        
        if request.return_sources and 'sources' in result:
            response.sources = [
                SourceDocument(**source) for source in result['sources']
            ]
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_rag_stream(request: QueryRequest):
    """
    Query the RAG system with streaming response
    
    Returns Server-Sent Events (SSE) stream
    """
    try:
        logger.info(f"Processing streaming query: {request.query[:100]}...")
        
        async def generate():
            async for chunk in await rag_pipeline.query(
                query=request.query,
                use_hybrid_search=request.use_hybrid_search,
                stream=True,
                return_sources=False
            ):
                yield f"data: {chunk}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Error in streaming query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_system(request: EvaluationRequest):
    """
    Evaluate RAG system using RAGAS and DeepEval
    
    Runs comprehensive evaluation and returns detailed metrics
    """
    try:
        logger.info(f"Starting evaluation with {len(request.questions)} examples...")
        
        # Run evaluation
        results = evaluator.evaluate_all(
            questions=request.questions,
            answers=request.answers,
            contexts=request.contexts,
            ground_truths=request.ground_truths,
            save_results=request.save_results
        )
        
        # Generate report
        full_report = evaluator.generate_full_report(results)
        
        return EvaluationResponse(
            timestamp=results['timestamp'],
            num_examples=results['num_examples'],
            ragas_overall_score=results['ragas']['overall_score'],
            deepeval_overall_score=results['deepeval']['overall_score'],
            average_score=results['comparison']['overall_scores']['average'],
            full_report=full_report
        )
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def clear_documents():
    """
    Clear all documents from the vector store
    
    Use with caution - this deletes all indexed documents
    """
    try:
        logger.info("Clearing all documents...")
        await rag_pipeline.clear_all_documents()
        
        return {"message": "All documents cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    Get system statistics
    
    Returns configuration and status information
    """
    return {
        "config": {
            "vector_db": settings.vector_db,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "rerank_top_k": settings.rerank_top_k,
            "guardrails_enabled": settings.enable_guardrails,
            "toxicity_threshold": settings.toxicity_threshold
        },
        "pipeline": {
            "chunking_strategy": rag_pipeline.chunking_strategy,
            "reranker_type": rag_pipeline.reranker_type
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
