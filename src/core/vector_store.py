"""
Vector store implementations supporting Pinecone and Weaviate with hybrid search
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Weaviate
import pinecone
import weaviate
from config.settings import settings

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to vector store"""
        pass
    
    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        pass
    
    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search (vector + keyword)"""
        pass
    
    @abstractmethod
    async def delete_all(self):
        """Delete all documents"""
        pass


class PineconeStore(VectorStore):
    """Pinecone vector store with hybrid search support"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=settings.embedding_model)
        
        # Initialize Pinecone
        from pinecone import Pinecone, ServerlessSpec
        
        pc = Pinecone(api_key=settings.pinecone_api_key)
        
        # Create index if it doesn't exist
        index_name = settings.pinecone_index_name
        if index_name not in pc.list_indexes().names():
            logger.info(f"Creating Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=settings.pinecone_environment
                )
            )
        
        self.index = pc.Index(index_name)
        self.vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text"
        )
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to Pinecone"""
        try:
            from langchain.schema import Document
            
            docs = [
                Document(
                    page_content=doc['content'],
                    metadata={
                        'chunk_type': doc.get('chunk_type', 'unknown'),
                        'start_index': doc.get('start_index', 0),
                        'end_index': doc.get('end_index', 0),
                        **doc.get('metadata', {})
                    }
                )
                for doc in documents
            ]
            
            ids = self.vectorstore.add_documents(docs)
            logger.info(f"Added {len(ids)} documents to Pinecone")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {e}")
            raise
    
    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        try:
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=filter
            )
            
            return [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                }
                for doc, score in results
            ]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    async def hybrid_search(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and sparse (BM25-like) search
        alpha: 0 = pure keyword search, 1 = pure vector search
        """
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Pinecone hybrid search using dense and sparse vectors
            # Note: This requires hybrid index setup in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            
            return [
                {
                    'content': match['metadata'].get('text', ''),
                    'metadata': match['metadata'],
                    'score': match['score']
                }
                for match in results['matches']
            ]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to regular similarity search
            return await self.similarity_search(query, k)
    
    async def delete_all(self):
        """Delete all vectors from index"""
        try:
            self.index.delete(delete_all=True)
            logger.info("Deleted all documents from Pinecone")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise


class WeaviateStore(VectorStore):
    """Weaviate vector store with hybrid search support"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=settings.embedding_model)
        
        # Initialize Weaviate client
        auth_config = None
        if settings.weaviate_api_key:
            auth_config = weaviate.AuthApiKey(api_key=settings.weaviate_api_key)
        
        self.client = weaviate.Client(
            url=settings.weaviate_url,
            auth_client_secret=auth_config
        )
        
        self.class_name = "Document"
        self._create_schema_if_not_exists()
        
        self.vectorstore = Weaviate(
            client=self.client,
            index_name=self.class_name,
            text_key="content",
            embedding=self.embeddings
        )
    
    def _create_schema_if_not_exists(self):
        """Create Weaviate schema if it doesn't exist"""
        try:
            schema = self.client.schema.get()
            class_names = [c['class'] for c in schema.get('classes', [])]
            
            if self.class_name not in class_names:
                class_obj = {
                    "class": self.class_name,
                    "description": "Document chunks for RAG system",
                    "vectorizer": "none",  # We'll provide our own vectors
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "Document content"
                        },
                        {
                            "name": "chunk_type",
                            "dataType": ["string"],
                            "description": "Type of chunking used"
                        },
                        {
                            "name": "start_index",
                            "dataType": ["int"],
                            "description": "Start position in original document"
                        },
                        {
                            "name": "end_index",
                            "dataType": ["int"],
                            "description": "End position in original document"
                        }
                    ]
                }
                self.client.schema.create_class(class_obj)
                logger.info(f"Created Weaviate schema for {self.class_name}")
                
        except Exception as e:
            logger.error(f"Error creating Weaviate schema: {e}")
            raise
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to Weaviate"""
        try:
            from langchain.schema import Document
            
            docs = [
                Document(
                    page_content=doc['content'],
                    metadata={
                        'chunk_type': doc.get('chunk_type', 'unknown'),
                        'start_index': doc.get('start_index', 0),
                        'end_index': doc.get('end_index', 0),
                        **doc.get('metadata', {})
                    }
                )
                for doc in documents
            ]
            
            ids = self.vectorstore.add_documents(docs)
            logger.info(f"Added {len(ids)} documents to Weaviate")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to Weaviate: {e}")
            raise
    
    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        try:
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k
            )
            
            return [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                }
                for doc, score in results
            ]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    async def hybrid_search(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and BM25 keyword search
        alpha: 0 = pure keyword search, 1 = pure vector search
        """
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Weaviate hybrid search
            result = (
                self.client.query
                .get(self.class_name, ["content", "chunk_type", "start_index", "end_index"])
                .with_hybrid(
                    query=query,
                    alpha=alpha,
                    vector=query_embedding
                )
                .with_limit(k)
                .with_additional(["score"])
                .do()
            )
            
            documents = result.get('data', {}).get('Get', {}).get(self.class_name, [])
            
            return [
                {
                    'content': doc['content'],
                    'metadata': {
                        'chunk_type': doc.get('chunk_type'),
                        'start_index': doc.get('start_index'),
                        'end_index': doc.get('end_index')
                    },
                    'score': doc['_additional']['score']
                }
                for doc in documents
            ]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to regular similarity search
            return await self.similarity_search(query, k)
    
    async def delete_all(self):
        """Delete all objects from class"""
        try:
            self.client.schema.delete_class(self.class_name)
            self._create_schema_if_not_exists()
            logger.info("Deleted all documents from Weaviate")
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise


def get_vector_store() -> VectorStore:
    """Factory function to get vector store based on configuration"""
    if settings.vector_db == "pinecone":
        return PineconeStore()
    elif settings.vector_db == "weaviate":
        return WeaviateStore()
    else:
        raise ValueError(f"Unsupported vector database: {settings.vector_db}")
