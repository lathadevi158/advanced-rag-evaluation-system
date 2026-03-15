"""
Guardrails for content safety, toxicity, and bias detection
"""
from typing import Dict, Any, List
import logging
from detoxify import Detoxify
from config.settings import settings

logger = logging.getLogger(__name__)


class ContentGuardrails:
    """
    Guardrails for checking content safety before processing
    Detects: toxicity, severe toxicity, obscenity, threats, insults, identity attacks
    """
    
    def __init__(self, toxicity_threshold: float = None):
        """
        Initialize content guardrails
        
        Args:
            toxicity_threshold: Threshold for toxicity (0-1), higher means stricter
        """
        self.toxicity_threshold = toxicity_threshold or settings.toxicity_threshold
        logger.info("Loading Detoxify model for content safety...")
        self.model = Detoxify('original')
        logger.info("Detoxify model loaded successfully")
    
    def check_content(self, text: str) -> Dict[str, Any]:
        """
        Check content for safety issues
        
        Args:
            text: Text to check
        
        Returns:
            Dictionary with safety scores and pass/fail status
        """
        try:
            # Get predictions from Detoxify
            predictions = self.model.predict(text)
            
            # Check if any category exceeds threshold
            violations = []
            for category, score in predictions.items():
                if score > self.toxicity_threshold:
                    violations.append({
                        'category': category,
                        'score': float(score),
                        'threshold': self.toxicity_threshold
                    })
            
            is_safe = len(violations) == 0
            
            result = {
                'is_safe': is_safe,
                'scores': {k: float(v) for k, v in predictions.items()},
                'violations': violations,
                'max_score': float(max(predictions.values())),
                'threshold': self.toxicity_threshold
            }
            
            if not is_safe:
                logger.warning(
                    f"Content safety check failed. Violations: {violations}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in content safety check: {e}")
            # Fail safe: assume content is unsafe if check fails
            return {
                'is_safe': False,
                'error': str(e),
                'scores': {},
                'violations': [{'category': 'error', 'score': 1.0}]
            }
    
    def check_query(self, query: str) -> Dict[str, Any]:
        """Check if user query is safe to process"""
        return self.check_content(query)
    
    def check_response(self, response: str) -> Dict[str, Any]:
        """Check if generated response is safe to return"""
        return self.check_content(response)


class BiasDetector:
    """
    Simple bias detection using keyword matching
    Can be extended with more sophisticated models
    """
    
    def __init__(self):
        # Categories of potentially biased language
        self.bias_keywords = {
            'gender': [
                'he should', 'she should', 'men are', 'women are',
                'boys are', 'girls are', 'male workers', 'female workers'
            ],
            'age': [
                'old people', 'young people', 'elderly', 'millennials',
                'boomers', 'too old', 'too young'
            ],
            'race': [
                'those people', 'they all', 'typical'
            ],
            'ability': [
                'handicapped', 'disabled people', 'normal people', 'suffers from'
            ]
        }
    
    def detect_bias(self, text: str) -> Dict[str, Any]:
        """
        Detect potential bias in text
        
        Args:
            text: Text to check
        
        Returns:
            Dictionary with bias detection results
        """
        text_lower = text.lower()
        detected_biases = []
        
        for category, keywords in self.bias_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_biases.append({
                        'category': category,
                        'keyword': keyword,
                        'severity': 'medium'  # Could be enhanced with severity scoring
                    })
        
        return {
            'has_bias': len(detected_biases) > 0,
            'biases': detected_biases,
            'bias_count': len(detected_biases)
        }


class GuardrailsPipeline:
    """
    Combined guardrails pipeline for comprehensive content checking
    """
    
    def __init__(self, enable_guardrails: bool = None):
        """
        Initialize guardrails pipeline
        
        Args:
            enable_guardrails: Whether to enable guardrails (from settings if None)
        """
        self.enabled = enable_guardrails if enable_guardrails is not None else settings.enable_guardrails
        
        if self.enabled:
            self.content_guardrails = ContentGuardrails()
            self.bias_detector = BiasDetector()
            logger.info("Guardrails pipeline initialized and enabled")
        else:
            logger.info("Guardrails pipeline disabled")
    
    def check_input(self, query: str) -> Dict[str, Any]:
        """
        Check user input before processing
        
        Args:
            query: User query
        
        Returns:
            Dictionary with all safety checks
        """
        if not self.enabled:
            return {'is_safe': True, 'enabled': False}
        
        safety_check = self.content_guardrails.check_query(query)
        bias_check = self.bias_detector.detect_bias(query)
        
        return {
            'is_safe': safety_check['is_safe'] and not bias_check['has_bias'],
            'safety': safety_check,
            'bias': bias_check,
            'enabled': True
        }
    
    def check_output(self, response: str) -> Dict[str, Any]:
        """
        Check generated response before returning to user
        
        Args:
            response: Generated response
        
        Returns:
            Dictionary with all safety checks
        """
        if not self.enabled:
            return {'is_safe': True, 'enabled': False}
        
        safety_check = self.content_guardrails.check_response(response)
        bias_check = self.bias_detector.detect_bias(response)
        
        return {
            'is_safe': safety_check['is_safe'] and not bias_check['has_bias'],
            'safety': safety_check,
            'bias': bias_check,
            'enabled': True
        }
    
    def get_safe_response_message(self, check_result: Dict[str, Any]) -> str:
        """
        Get user-friendly message when content fails safety checks
        
        Args:
            check_result: Result from check_output
        
        Returns:
            User-friendly error message
        """
        if check_result.get('safety', {}).get('violations'):
            return (
                "I apologize, but I cannot provide a response to this query as it may contain "
                "inappropriate or harmful content. Please rephrase your question."
            )
        
        if check_result.get('bias', {}).get('has_bias'):
            return (
                "I've detected potentially biased language in the response. "
                "Let me rephrase that more neutrally."
            )
        
        return "I apologize, but I cannot process this request due to content safety concerns."


# Global guardrails instance
guardrails = GuardrailsPipeline()
