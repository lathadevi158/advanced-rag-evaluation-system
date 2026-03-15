"""
DeepEval evaluation framework
Additional metrics for comprehensive RAG evaluation
"""
from typing import List, Dict, Any
import logging
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric
)
from deepeval.test_case import LLMTestCase
import pandas as pd

logger = logging.getLogger(__name__)


class DeepEvalEvaluator:
    """
    Evaluate RAG system using DeepEval metrics
    Provides complementary evaluation to RAGAS
    """
    
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        """
        Initialize DeepEval evaluator
        
        Args:
            model: LLM model to use for evaluation
        """
        self.model = model
        
        # Initialize metrics
        self.answer_relevancy = AnswerRelevancyMetric(
            threshold=0.7,
            model=model
        )
        
        self.faithfulness = FaithfulnessMetric(
            threshold=0.7,
            model=model
        )
        
        self.contextual_precision = ContextualPrecisionMetric(
            threshold=0.7,
            model=model
        )
        
        self.contextual_recall = ContextualRecallMetric(
            threshold=0.7,
            model=model
        )
        
        self.hallucination = HallucinationMetric(
            threshold=0.5,
            model=model
        )
        
        self.metrics = [
            self.answer_relevancy,
            self.faithfulness,
            self.contextual_precision,
            self.contextual_recall,
            self.hallucination
        ]
        
        logger.info(f"DeepEval evaluator initialized with model: {model}")
    
    def create_test_case(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> LLMTestCase:
        """
        Create DeepEval test case
        
        Args:
            question: User question (input)
            answer: Generated answer (actual_output)
            contexts: Retrieved contexts (retrieval_context)
            ground_truth: Expected answer (expected_output)
        
        Returns:
            LLMTestCase for evaluation
        """
        return LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth,
            retrieval_context=contexts
        )
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single test case
        
        Args:
            question: User question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Expected answer
        
        Returns:
            Evaluation results for all metrics
        """
        test_case = self.create_test_case(question, answer, contexts, ground_truth)
        
        results = {}
        
        try:
            # Evaluate with each metric
            for metric in self.metrics:
                metric.measure(test_case)
                metric_name = metric.__class__.__name__.replace('Metric', '').lower()
                results[metric_name] = {
                    'score': metric.score,
                    'reason': metric.reason,
                    'success': metric.is_successful()
                }
            
            # Calculate overall score
            scores = [r['score'] for r in results.values() if r['score'] is not None]
            results['overall_score'] = sum(scores) / len(scores) if scores else 0.0
            
            logger.info(f"Single evaluation complete. Overall: {results['overall_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Error in DeepEval evaluation: {e}")
            raise
        
        return results
    
    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate multiple test cases
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved contexts
            ground_truths: List of expected answers
        
        Returns:
            Aggregated evaluation results
        """
        logger.info(f"Evaluating {len(questions)} examples with DeepEval...")
        
        all_results = []
        
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
            try:
                result = self.evaluate_single(q, a, c, gt)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate example: {e}")
                continue
        
        # Aggregate results
        if not all_results:
            raise ValueError("No successful evaluations")
        
        aggregated = self._aggregate_results(all_results)
        aggregated['num_examples'] = len(all_results)
        
        logger.info(f"Batch evaluation complete. Overall: {aggregated['overall_score']:.4f}")
        
        return aggregated
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple evaluations"""
        metric_names = [
            'answer_relevancy',
            'faithfulness',
            'contextual_precision',
            'contextual_recall',
            'hallucination'
        ]
        
        aggregated = {}
        
        for metric_name in metric_names:
            scores = [
                r[metric_name]['score']
                for r in results
                if metric_name in r and r[metric_name]['score'] is not None
            ]
            
            if scores:
                aggregated[metric_name] = {
                    'mean': sum(scores) / len(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }
        
        # Overall score
        overall_scores = [r['overall_score'] for r in results]
        aggregated['overall_score'] = sum(overall_scores) / len(overall_scores)
        
        return aggregated
    
    def get_detailed_report(
        self,
        evaluation_result: Dict[str, Any]
    ) -> str:
        """
        Generate detailed evaluation report
        
        Args:
            evaluation_result: Result from evaluate_batch()
        
        Returns:
            Formatted report string
        """
        overall = evaluation_result['overall_score']
        num_examples = evaluation_result.get('num_examples', 0)
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                  DeepEval Evaluation Report                   ║
╚══════════════════════════════════════════════════════════════╝

Overall Score: {overall:.4f}
Examples Evaluated: {num_examples}

Detailed Metrics:
─────────────────────────────────────────────────────────────────
"""
        
        metrics = {
            'answer_relevancy': 'Answer Relevancy',
            'faithfulness': 'Faithfulness',
            'contextual_precision': 'Contextual Precision',
            'contextual_recall': 'Contextual Recall',
            'hallucination': 'Hallucination Score'
        }
        
        for key, name in metrics.items():
            if key in evaluation_result:
                metric_data = evaluation_result[key]
                report += f"""
{name}:
  Mean:  {metric_data['mean']:.4f}
  Range: {metric_data['min']:.4f} - {metric_data['max']:.4f}
"""
        
        report += "\n─────────────────────────────────────────────────────────────────\n"
        
        return report
    
    def compare_with_ragas(
        self,
        deepeval_results: Dict[str, Any],
        ragas_results: Dict[str, Any]
    ) -> str:
        """
        Compare DeepEval and RAGAS results
        
        Args:
            deepeval_results: Results from DeepEval
            ragas_results: Results from RAGAS
        
        Returns:
            Comparison report
        """
        report = """
╔══════════════════════════════════════════════════════════════╗
║              DeepEval vs RAGAS Comparison                     ║
╚══════════════════════════════════════════════════════════════╝

Metric Comparison:
─────────────────────────────────────────────────────────────────
"""
        
        # Compare overlapping metrics
        comparisons = {
            'Faithfulness': (
                ragas_results['metrics'].get('faithfulness', 0),
                deepeval_results.get('faithfulness', {}).get('mean', 0)
            ),
            'Answer Relevancy': (
                ragas_results['metrics'].get('answer_relevancy', 0),
                deepeval_results.get('answer_relevancy', {}).get('mean', 0)
            ),
            'Context Precision': (
                ragas_results['metrics'].get('context_precision', 0),
                deepeval_results.get('contextual_precision', {}).get('mean', 0)
            ),
            'Context Recall': (
                ragas_results['metrics'].get('context_recall', 0),
                deepeval_results.get('contextual_recall', {}).get('mean', 0)
            )
        }
        
        for metric_name, (ragas_score, deepeval_score) in comparisons.items():
            diff = abs(ragas_score - deepeval_score)
            report += f"""
{metric_name}:
  RAGAS:    {ragas_score:.4f}
  DeepEval: {deepeval_score:.4f}
  Diff:     {diff:.4f}
"""
        
        report += f"""
─────────────────────────────────────────────────────────────────
Overall Scores:
  RAGAS:    {ragas_results['overall_score']:.4f}
  DeepEval: {deepeval_results['overall_score']:.4f}
"""
        
        return report


def create_deepeval_evaluator(model: str = "gpt-4-turbo-preview") -> DeepEvalEvaluator:
    """Factory function to create DeepEval evaluator"""
    return DeepEvalEvaluator(model=model)
