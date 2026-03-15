"""
Advanced chunking strategies: Semantic and Fixed-size chunking
"""
import re
from typing import List, Dict, Any
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import numpy as np
from config.settings import settings


class SemanticChunker:
    """
    Semantic chunker that groups text based on semantic similarity
    using cosine similarity between embeddings
    """
    
    def __init__(
        self,
        embedding_model: str = None,
        breakpoint_threshold: float = 0.5,
        max_chunk_size: int = 512
    ):
        self.embedding_model = embedding_model or settings.embedding_model
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.breakpoint_threshold = breakpoint_threshold
        self.max_chunk_size = max_chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        sentence_endings = r'[.!?]+[\s]+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def _get_token_count(self, text: str) -> int:
        """Get token count for text"""
        return len(self.tokenizer.encode(text))
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text semantically by grouping similar sentences
        
        Returns:
            List of dictionaries with 'content', 'start_index', 'end_index'
        """
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [{
                'content': text,
                'start_index': 0,
                'end_index': len(text),
                'chunk_type': 'semantic'
            }]
        
        # Get embeddings for all sentences
        sentence_embeddings = self.embeddings.embed_documents(sentences)
        
        # Calculate similarities between consecutive sentences
        similarities = []
        for i in range(len(sentence_embeddings) - 1):
            sim = self._calculate_similarity(
                sentence_embeddings[i],
                sentence_embeddings[i + 1]
            )
            similarities.append(sim)
        
        # Find breakpoints where similarity drops below threshold
        chunks = []
        current_chunk = [sentences[0]]
        current_position = 0
        
        for i, similarity in enumerate(similarities):
            current_chunk.append(sentences[i + 1])
            
            # Check if we should create a breakpoint
            chunk_text = ' '.join(current_chunk)
            token_count = self._get_token_count(chunk_text)
            
            if similarity < self.breakpoint_threshold or token_count > self.max_chunk_size:
                # Save current chunk
                chunk_content = ' '.join(current_chunk[:-1]) if similarity < self.breakpoint_threshold else chunk_text
                chunks.append({
                    'content': chunk_content,
                    'start_index': current_position,
                    'end_index': current_position + len(chunk_content),
                    'chunk_type': 'semantic',
                    'similarity_score': similarity if similarity < self.breakpoint_threshold else None
                })
                
                # Start new chunk
                current_chunk = [sentences[i + 1]] if similarity < self.breakpoint_threshold else []
                current_position += len(chunk_content) + 1
        
        # Add remaining chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append({
                'content': chunk_content,
                'start_index': current_position,
                'end_index': current_position + len(chunk_content),
                'chunk_type': 'semantic'
            })
        
        return chunks


class FixedSizeChunker:
    """Traditional fixed-size chunking with overlap"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text using fixed-size strategy
        
        Returns:
            List of dictionaries with 'content', 'start_index', 'end_index'
        """
        chunks = self.splitter.split_text(text)
        
        result = []
        current_position = 0
        
        for chunk in chunks:
            result.append({
                'content': chunk,
                'start_index': current_position,
                'end_index': current_position + len(chunk),
                'chunk_type': 'fixed_size'
            })
            current_position += len(chunk) - self.chunk_overlap
        
        return result


class HybridChunker:
    """
    Hybrid chunker that uses semantic chunking first, 
    then splits large chunks with fixed-size strategy
    """
    
    def __init__(
        self,
        max_chunk_size: int = 512,
        semantic_threshold: float = 0.5
    ):
        self.semantic_chunker = SemanticChunker(
            breakpoint_threshold=semantic_threshold,
            max_chunk_size=max_chunk_size
        )
        self.fixed_chunker = FixedSizeChunker(chunk_size=max_chunk_size)
    
    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """Apply semantic chunking with fixed-size fallback for large chunks"""
        semantic_chunks = self.semantic_chunker.chunk(text)
        
        final_chunks = []
        for chunk in semantic_chunks:
            # If chunk is too large, split it further
            if len(chunk['content']) > self.semantic_chunker.max_chunk_size * 1.5:
                sub_chunks = self.fixed_chunker.chunk(chunk['content'])
                for sub_chunk in sub_chunks:
                    sub_chunk['chunk_type'] = 'hybrid'
                    final_chunks.append(sub_chunk)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
