"""
Combined evaluation runner for comprehensive RAG assessment
Runs both RAGAS and DeepEval, generates comparison reports
"""
from typing import List, Dict, Any
import logging
import json
from datetime import datetime
import pandas as pd

from src.evaluation.ragas_eval import create_ragas_evaluator
from src.evaluation.deepeval_eval import create_deepeval_evaluator

logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """
    Run both RAGAS and DeepEval evaluations and compare results
    """
    
    def __init__(self, eval_model: str = "gpt-4-turbo-preview"):
        """
        Initialize comprehensive evaluator
        
        Args:
            eval_model: Model to use for LLM-based evaluation metrics
        """
        self.ragas_evaluator = create_ragas_evaluator()
        self.deepeval_evaluator = create_deepeval_evaluator(model=eval_model)
        logger.info("Comprehensive evaluator initialized")
    
    def evaluate_all(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
        save_results: bool = True,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation with both frameworks
        
        Args:
            questions: User questions
            answers: Generated answers
            contexts: Retrieved contexts
            ground_truths: Expected answers
            save_results: Whether to save results to file
            output_path: Path to save results (auto-generated if None)
        
        Returns:
            Combined evaluation results
        """
        logger.info(f"Starting comprehensive evaluation of {len(questions)} examples...")
        
        # Run RAGAS evaluation
        logger.info("Running RAGAS evaluation...")
        ragas_results = self.ragas_evaluator.evaluate(
            questions, answers, contexts, ground_truths
        )
        
        # Run DeepEval evaluation
        logger.info("Running DeepEval evaluation...")
        deepeval_results = self.deepeval_evaluator.evaluate_batch(
            questions, answers, contexts, ground_truths
        )
        
        # Combine results
        combined_results = {
            'timestamp': datetime.now().isoformat(),
            'num_examples': len(questions),
            'ragas': ragas_results,
            'deepeval': deepeval_results,
            'comparison': self._compare_results(ragas_results, deepeval_results)
        }
        
        # Save if requested
        if save_results:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"data/evaluation_results_{timestamp}.json"
            
            self._save_results(combined_results, output_path)
        
        logger.info("Comprehensive evaluation complete")
        
        return combined_results
    
    def _compare_results(
        self,
        ragas_results: Dict[str, Any],
        deepeval_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare RAGAS and DeepEval results"""
        
        comparison = {
            'overall_scores': {
                'ragas': ragas_results['overall_score'],
                'deepeval': deepeval_results['overall_score'],
                'average': (ragas_results['overall_score'] + deepeval_results['overall_score']) / 2
            },
            'metric_comparison': {}
        }
        
        # Compare overlapping metrics
        metric_mapping = {
            'faithfulness': 'faithfulness',
            'answer_relevancy': 'answer_relevancy',
            'context_precision': 'contextual_precision',
            'context_recall': 'contextual_recall'
        }
        
        for ragas_key, deepeval_key in metric_mapping.items():
            ragas_score = ragas_results['metrics'].get(ragas_key, 0)
            deepeval_score = deepeval_results.get(deepeval_key, {}).get('mean', 0)
            
            comparison['metric_comparison'][ragas_key] = {
                'ragas': ragas_score,
                'deepeval': deepeval_score,
                'difference': abs(ragas_score - deepeval_score),
                'average': (ragas_score + deepeval_score) / 2
            }
        
        return comparison
    
    def _save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to JSON file"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def generate_full_report(
        self,
        evaluation_results: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            evaluation_results: Results from evaluate_all()
        
        Returns:
            Full formatted report
        """
        ragas_report = self.ragas_evaluator.get_detailed_report(
            evaluation_results['ragas']
        )
        
        deepeval_report = self.deepeval_evaluator.get_detailed_report(
            evaluation_results['deepeval']
        )
        
        comparison_report = self.deepeval_evaluator.compare_with_ragas(
            evaluation_results['deepeval'],
            evaluation_results['ragas']
        )
        
        full_report = f"""
{'='*70}
                    COMPREHENSIVE RAG EVALUATION
{'='*70}

Timestamp: {evaluation_results['timestamp']}
Examples Evaluated: {evaluation_results['num_examples']}

{ragas_report}

{deepeval_report}

{comparison_report}

{'='*70}
                         RECOMMENDATIONS
{'='*70}

"""
        
        # Add recommendations based on scores
        comparison = evaluation_results['comparison']
        overall_avg = comparison['overall_scores']['average']
        
        if overall_avg >= 0.8:
            full_report += "✅ Excellent performance across all metrics!\n"
        elif overall_avg >= 0.6:
            full_report += "⚠️  Good performance with room for improvement.\n"
        else:
            full_report += "❌ Performance needs significant improvement.\n"
        
        # Specific recommendations
        metrics = comparison['metric_comparison']
        
        if metrics.get('faithfulness', {}).get('average', 1) < 0.7:
            full_report += "\n→ Focus on reducing hallucinations and improving faithfulness.\n"
        
        if metrics.get('context_recall', {}).get('average', 1) < 0.7:
            full_report += "\n→ Consider improving retrieval to get more relevant context.\n"
        
        if metrics.get('answer_relevancy', {}).get('average', 1) < 0.7:
            full_report += "\n→ Improve prompt engineering to generate more relevant answers.\n"
        
        full_report += f"\n{'='*70}\n"
        
        return full_report
    
    def evaluate_and_report(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
        save_results: bool = True
    ) -> str:
        """
        Convenience method: evaluate and generate report in one call
        
        Args:
            questions: User questions
            answers: Generated answers
            contexts: Retrieved contexts
            ground_truths: Expected answers
            save_results: Whether to save results
        
        Returns:
            Full evaluation report
        """
        results = self.evaluate_all(
            questions, answers, contexts, ground_truths, save_results
        )
        
        return self.generate_full_report(results)


def create_comprehensive_evaluator(
    eval_model: str = "gpt-4-turbo-preview"
) -> ComprehensiveEvaluator:
    """Factory function to create comprehensive evaluator"""
    return ComprehensiveEvaluator(eval_model=eval_model)
