import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core import create_rag_pipeline
from src.evaluation import create_comprehensive_evaluator


async def example_basic_usage():
    """Basic RAG pipeline usage"""
    print("="*70)
    print("EXAMPLE 1: Basic RAG Pipeline Usage")
    print("="*70)
    
    # Create pipeline
    pipeline = create_rag_pipeline(
        chunking_strategy="semantic",
        reranker_type="cross_encoder"
    )
    
    # Sample documents
    documents = [
        """
        Artificial Intelligence (AI) is transforming healthcare through various applications.
        Machine learning algorithms can analyze medical images to detect diseases like cancer
        with high accuracy. AI-powered systems assist doctors in diagnosing conditions,
        predicting patient outcomes, and personalizing treatment plans.
        """,
        """
        Natural Language Processing (NLP) enables computers to understand and generate human language.
        Applications include chatbots, translation services, sentiment analysis, and text summarization.
        Recent advances in transformer models like GPT and BERT have significantly improved NLP capabilities.
        """,
        """
        Computer vision allows machines to interpret and understand visual information.
        Applications range from facial recognition and autonomous vehicles to medical imaging
        and quality control in manufacturing. Deep learning has revolutionized computer vision
        accuracy in recent years.
        """
    ]
    
    # Ingest documents
    print("\n📥 Ingesting documents...")
    stats = await pipeline.ingest_documents(documents)
    print(f"✅ Ingested {stats['total_chunks']} chunks from {stats['total_documents']} documents")
    print(f"   Average chunk size: {stats['avg_chunk_size']:.0f} characters")
    
    # Query the system
    print("\n🔍 Querying the system...")
    query = "How is AI being used in healthcare?"
    
    result = await pipeline.query(
        query=query,
        use_hybrid_search=True,
        stream=False,
        return_sources=True
    )
    
    print(f"\n❓ Question: {query}")
    print(f"\n💡 Answer: {result['answer']}")
    print(f"\n📚 Sources used: {result['num_sources']}")
    
    if result.get('sources'):
        print("\nSource documents:")
        for idx, source in enumerate(result['sources'], 1):
            print(f"  {idx}. Score: {source['score']:.4f}")
            print(f"     {source['content'][:100]}...")
    
    # Cleanup
    await pipeline.clear_all_documents()
    print("\n✨ Example complete!\n")


async def example_streaming():
    """Example with streaming response"""
    print("="*70)
    print("EXAMPLE 2: Streaming Response")
    print("="*70)
    
    pipeline = create_rag_pipeline()
    
    # Ingest a document
    documents = [
        """
        Python is a high-level, interpreted programming language known for its simplicity
        and readability. It supports multiple programming paradigms including procedural,
        object-oriented, and functional programming. Python's extensive standard library
        and vast ecosystem of third-party packages make it suitable for web development,
        data science, machine learning, automation, and more.
        """
    ]
    
    await pipeline.ingest_documents(documents)
    
    print("\n🔍 Querying with streaming...")
    query = "What is Python used for?"
    print(f"❓ Question: {query}")
    print(f"\n💡 Answer: ", end="", flush=True)
    
    # Stream the response
    async for chunk in await pipeline.query(
        query=query,
        stream=True,
        return_sources=False
    ):
        print(chunk, end="", flush=True)
    
    print("\n\n✨ Streaming example complete!\n")
    
    await pipeline.clear_all_documents()


async def example_evaluation():
    """Example evaluation with RAGAS and DeepEval"""
    print("="*70)
    print("EXAMPLE 3: Comprehensive Evaluation")
    print("="*70)
    
    pipeline = create_rag_pipeline()
    evaluator = create_comprehensive_evaluator()
    
    # Sample documents
    documents = [
        "The Eiffel Tower is located in Paris, France. It was completed in 1889.",
        "The Great Wall of China is a series of fortifications built across northern China."
    ]
    
    await pipeline.ingest_documents(documents)
    
    # Prepare evaluation data
    questions = [
        "Where is the Eiffel Tower located?",
        "When was the Eiffel Tower completed?"
    ]
    
    ground_truths = [
        "The Eiffel Tower is located in Paris, France.",
        "The Eiffel Tower was completed in 1889."
    ]
    
    # Generate answers and get contexts
    answers = []
    contexts = []
    
    for question in questions:
        result = await pipeline.query(
            query=question,
            return_sources=True
        )
        answers.append(result['answer'])
        contexts.append([src['content'] for src in result.get('sources', [])])
    
    print("\n📊 Running comprehensive evaluation...")
    
    # Run evaluation
    report = evaluator.evaluate_and_report(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
        save_results=False
    )
    
    print(report)
    
    await pipeline.clear_all_documents()
    print("✨ Evaluation example complete!\n")


async def example_chunking_comparison():
    """Compare different chunking strategies"""
    print("="*70)
    print("EXAMPLE 4: Chunking Strategy Comparison")
    print("="*70)
    
    document = """
    Machine learning is a subset of artificial intelligence that enables systems to learn
    from data without explicit programming. It uses algorithms to identify patterns and
    make predictions. There are three main types: supervised learning, unsupervised learning,
    and reinforcement learning. Supervised learning uses labeled data to train models.
    Unsupervised learning finds patterns in unlabeled data. Reinforcement learning involves
    agents learning through trial and error with rewards and penalties.
    """
    
    strategies = ["semantic", "fixed", "hybrid"]
    
    for strategy in strategies:
        print(f"\n📐 Testing {strategy.upper()} chunking:")
        pipeline = create_rag_pipeline(chunking_strategy=strategy)
        
        stats = await pipeline.ingest_documents([document])
        
        print(f"  Chunks created: {stats['total_chunks']}")
        print(f"  Avg chunk size: {stats['avg_chunk_size']:.0f} chars")
        print(f"  Min: {stats['min_chunk_size']}, Max: {stats['max_chunk_size']}")
        
        await pipeline.clear_all_documents()
    
    print("\n✨ Chunking comparison complete!\n")


async def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("  PRODUCTION RAG SYSTEM - EXAMPLES")
    print("="*70 + "\n")
    
    try:
        await example_basic_usage()
        await example_streaming()
        await example_chunking_comparison()
        
        # Note: Evaluation example requires API keys for GPT-4
        # Uncomment if you have the keys configured
        # await example_evaluation()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
