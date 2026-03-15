"""
RAGAS (Retrieval-Augmented Generation Assessment) evaluation
Metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy
"""
from typing import List, Dict, Any
import logging
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    context_relevancy
)
from datasets import Dataset
import pandas as pd

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    """
    Evaluate RAG system using RAGAS metrics
    """
    
    def __init__(self):
        """Initialize RAGAS evaluator with all metrics"""
        self.metrics = [
            context_precision,      # How many relevant chunks in top-k?
            context_recall,         # Did we retrieve all relevant info?
            faithfulness,          # Is answer grounded in context?
            answer_relevancy,      # Does answer address the question?
            context_relevancy      # Is retrieved context relevant?
        ]
        logger.info("RAGAS evaluator initialized with all metrics")
    
    def prepare_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> Dataset:
        """
        Prepare dataset for RAGAS evaluation
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved contexts (list of chunks per question)
            ground_truths: List of ground truth answers
        
        Returns:
            Hugging Face Dataset
        """
        data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths
        }
        
        return Dataset.from_dict(data)
    
    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system using RAGAS
        
        Args:
            questions: User questions
            answers: Generated answers
            contexts: Retrieved contexts for each question
            ground_truths: Expected answers
        
        Returns:
            Evaluation results with all metrics
        """
        logger.info(f"Evaluating {len(questions)} examples with RAGAS...")
        
        try:
            # Prepare dataset
            dataset = self.prepare_dataset(
                questions,
                answers,
                contexts,
                ground_truths
            )
            
            # Run evaluation
            result = evaluate(
                dataset,
                metrics=self.metrics
            )
            
            # Convert to dict and add summary
            scores = result.to_pandas().to_dict('records')[0]
            
            summary = {
                'overall_score': sum(scores.values()) / len(scores),
                'metrics': scores,
                'num_examples': len(questions)
            }
            
            logger.info(f"RAGAS evaluation complete. Overall score: {summary['overall_score']:.4f}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in RAGAS evaluation: {e}")
            raise
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> Dict[str, Any]:
        """Evaluate a single QA pair"""
        return self.evaluate(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth]
        )
    
    def get_detailed_report(
        self,
        evaluation_result: Dict[str, Any]
    ) -> str:
        """
        Generate detailed evaluation report
        
        Args:
            evaluation_result: Result from evaluate()
        
        Returns:
            Formatted report string
        """
        metrics = evaluation_result['metrics']
        overall = evaluation_result['overall_score']
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    RAGAS Evaluation Report                    ║
╚══════════════════════════════════════════════════════════════╝

Overall Score: {overall:.4f}

Detailed Metrics:
─────────────────────────────────────────────────────────────────

1. Context Precision: {metrics.get('context_precision', 0):.4f}
   → Measures how many of the retrieved chunks are actually relevant
   → Higher is better (1.0 = all retrieved chunks are relevant)

2. Context Recall: {metrics.get('context_recall', 0):.4f}
   → Measures if all relevant information was retrieved
   → Higher is better (1.0 = retrieved all needed information)

3. Faithfulness: {metrics.get('faithfulness', 0):.4f}
   → Measures if answer is grounded in retrieved context
   → Higher is better (1.0 = no hallucinations)

4. Answer Relevancy: {metrics.get('answer_relevancy', 0):.4f}
   → Measures if answer addresses the question
   → Higher is better (1.0 = perfectly relevant)

5. Context Relevancy: {metrics.get('context_relevancy', 0):.4f}
   → Measures relevance of retrieved context to question
   → Higher is better (1.0 = all context is relevant)

─────────────────────────────────────────────────────────────────
Examples Evaluated: {evaluation_result['num_examples']}
"""
        return report


def create_ragas_evaluator() -> RAGASEvaluator:
    """Factory function to create RAGAS evaluator"""
    return RAGASEvaluator()
