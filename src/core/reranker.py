"""
Cross-encoder reranker for improving retrieval quality
"""
from typing import List, Dict, Any
import logging
from sentence_transformers import CrossEncoder
import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranker using cross-encoder model to score query-document pairs
    More accurate than bi-encoder but slower (used for final reranking)
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker
        
        Args:
            model_name: Hugging Face model name for cross-encoder
                       Default: ms-marco-MiniLM-L-6-v2 (fast and accurate)
                       Alternatives:
                       - cross-encoder/ms-marco-MiniLM-L-12-v2 (more accurate, slower)
                       - cross-encoder/ms-marco-TinyBERT-L-2-v2 (fastest, less accurate)
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on cross-encoder scores
        
        Args:
            query: Search query
            documents: List of document dictionaries with 'content' and 'metadata'
            top_k: Number of top documents to return
        
        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, doc['content']] for doc in documents]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Add rerank scores to documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
            doc['original_score'] = doc.get('score', 0.0)
        
        # Sort by rerank score
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(
            f"Reranked {len(documents)} documents. "
            f"Top score: {reranked[0]['rerank_score']:.4f}, "
            f"Bottom score: {reranked[-1]['rerank_score']:.4f}"
        )
        
        return reranked[:top_k]


class HybridReranker:
    """
    Combines multiple reranking signals:
    1. Cross-encoder score
    2. Original retrieval score
    3. Diversity (MMR-like)
    """
    
    def __init__(
        self,
        cross_encoder_weight: float = 0.7,
        retrieval_weight: float = 0.3,
        diversity_penalty: float = 0.1
    ):
        """
        Initialize hybrid reranker
        
        Args:
            cross_encoder_weight: Weight for cross-encoder score (0-1)
            retrieval_weight: Weight for original retrieval score (0-1)
            diversity_penalty: Penalty for similar documents (0-1)
        """
        self.cross_encoder = CrossEncoderReranker()
        self.cross_encoder_weight = cross_encoder_weight
        self.retrieval_weight = retrieval_weight
        self.diversity_penalty = diversity_penalty
    
    def _calculate_diversity_penalty(
        self,
        doc: str,
        selected_docs: List[str]
    ) -> float:
        """
        Calculate penalty for documents similar to already selected ones
        Simple overlap-based diversity measure
        """
        if not selected_docs:
            return 0.0
        
        doc_words = set(doc.lower().split())
        max_overlap = 0.0
        
        for selected in selected_docs:
            selected_words = set(selected.lower().split())
            overlap = len(doc_words & selected_words) / len(doc_words | selected_words)
            max_overlap = max(max_overlap, overlap)
        
        return max_overlap * self.diversity_penalty
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Rerank using hybrid approach with diversity
        
        Args:
            query: Search query
            documents: List of documents
            top_k: Number of documents to return
        
        Returns:
            Reranked documents
        """
        if not documents:
            return []
        
        # Get cross-encoder scores
        reranked = self.cross_encoder.rerank(query, documents, top_k=len(documents))
        
        # Normalize scores to 0-1 range
        ce_scores = [doc['rerank_score'] for doc in reranked]
        orig_scores = [doc.get('original_score', 0.0) for doc in reranked]
        
        ce_min, ce_max = min(ce_scores), max(ce_scores)
        orig_min, orig_max = min(orig_scores), max(orig_scores)
        
        # Avoid division by zero
        ce_range = ce_max - ce_min if ce_max != ce_min else 1
        orig_range = orig_max - orig_min if orig_max != orig_min else 1
        
        # Greedy MMR-like selection with diversity
        selected = []
        remaining = reranked.copy()
        
        while len(selected) < top_k and remaining:
            best_idx = -1
            best_score = -float('inf')
            
            for idx, doc in enumerate(remaining):
                # Normalize scores
                ce_norm = (doc['rerank_score'] - ce_min) / ce_range
                orig_norm = (doc['original_score'] - orig_min) / orig_range
                
                # Combine scores
                combined_score = (
                    self.cross_encoder_weight * ce_norm +
                    self.retrieval_weight * orig_norm
                )
                
                # Apply diversity penalty
                diversity_penalty = self._calculate_diversity_penalty(
                    doc['content'],
                    [s['content'] for s in selected]
                )
                
                final_score = combined_score - diversity_penalty
                
                if final_score > best_score:
                    best_score = final_score
                    best_idx = idx
            
            if best_idx >= 0:
                doc = remaining.pop(best_idx)
                doc['final_score'] = best_score
                selected.append(doc)
        
        logger.info(f"Hybrid reranking selected {len(selected)} diverse documents")
        
        return selected


def get_reranker(reranker_type: str = "cross_encoder") -> Any:
    """
    Factory function to get reranker
    
    Args:
        reranker_type: Type of reranker ('cross_encoder' or 'hybrid')
    
    Returns:
        Reranker instance
    """
    if reranker_type == "cross_encoder":
        return CrossEncoderReranker()
    elif reranker_type == "hybrid":
        return HybridReranker()
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")
