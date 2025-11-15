"""
Vector Store Module for NLP CRM System
Handles ChromaDB setup, embedding generation, and semantic retrieval.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from config import Config
from document_processor import DocumentProcessor
from hybrid_search import HybridSearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages ChromaDB vector store operations for the CRM system."""
    
    def __init__(self, persist_directory: str = None, collection_name: str = "crm_knowledge_base"):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory or Config.CHROMA_PERSIST_DIRECTORY
        self.collection_name = collection_name
        
        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = self._initialize_embedding_model()
        
        # Initialize ChromaDB client
        self.chroma_client = self._initialize_chroma_client()
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
        # Initialize hybrid search engine
        self.hybrid_engine = HybridSearchEngine()
        
        logger.info(f"Vector store manager initialized with collection: {collection_name}")
    
    def _initialize_embedding_model(self) -> HuggingFaceEmbeddings:
        """Initialize the embedding model for text vectorization."""
        try:
            # Use a lightweight, effective embedding model
            model_name = "all-MiniLM-L6-v2"  # Good balance of speed and quality
            
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info(f"Initialized embedding model: {model_name}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
    
    def _initialize_chroma_client(self) -> chromadb.ClientAPI:
        """Initialize ChromaDB client with persistent storage."""
        try:
            client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            logger.info(f"ChromaDB client initialized with persist directory: {self.persist_directory}")
            return client
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            raise
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize the LangChain Chroma vector store."""
        try:
            vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )
            
            logger.info(f"Vector store initialized with collection: {self.collection_name}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
        """
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return []
            
            # Add documents to vector store
            document_ids = self.vector_store.add_documents(documents)
            
            # Persist the changes
            self.vector_store.persist()
            
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            return document_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Dict[str, Any] = None) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional filter criteria
            
        Returns:
            List of similar documents
        """
        try:
            if filter_dict:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k
                )
            
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            logger.info(f"Found {len(results)} similar documents with scores for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()
            
            info = {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}
    
    def reset_collection(self):
        """Reset the collection (delete all documents)."""
        try:
            # Delete the collection
            self.chroma_client.delete_collection(self.collection_name)
            
            # Reinitialize the vector store
            self.vector_store = self._initialize_vector_store()
            
            logger.info(f"Collection {self.collection_name} has been reset")
            
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            raise
    
    def setup_knowledge_base(self, kb_file_path: str = "customer_service_kb.md") -> bool:
        """
        Set up the complete knowledge base from the customer service KB file.
        
        Args:
            kb_file_path: Path to the knowledge base file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Process the document
            processor = DocumentProcessor()
            documents = processor.process_customer_service_kb(kb_file_path)
            
            if not documents:
                logger.error("No documents were processed from the knowledge base file")
                return False
            
            # Add documents to vector store
            document_ids = self.add_documents(documents)
            
            if document_ids:
                # Build BM25 index for hybrid search
                self.hybrid_engine.build_bm25_index(documents)
                
                logger.info(f"Knowledge base setup completed successfully with {len(document_ids)} documents")
                return True
            else:
                logger.error("Failed to add documents to vector store")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up knowledge base: {str(e)}")
            return False
    
    def hybrid_search(self, query: str, k: int = 5, bm25_weight: float = 0.3, 
                     vector_weight: float = 0.7) -> List[tuple]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Search query
            k: Number of results to return
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector scores
            
        Returns:
            List of (document, score) tuples
        """
        try:
            results = self.hybrid_engine.hybrid_search(
                query=query,
                vector_store=self,
                k=k,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight
            )
            
            logger.info(f"Hybrid search found {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []
    
    def hybrid_search_with_reranking(self, query: str, hybrid_k: int = 10, 
                                   final_k: int = 3, bm25_weight: float = 0.3, 
                                   vector_weight: float = 0.7) -> List[tuple]:
        """
        Perform hybrid search with reranking for high-quality results.
        
        Args:
            query: Search query
            hybrid_k: Number of results from hybrid search
            final_k: Number of final results after reranking
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector scores
            
        Returns:
            List of final reranked (document, score) tuples
        """
        try:
            results = self.hybrid_engine.hybrid_search_with_reranking(
                query=query,
                vector_store=self,
                hybrid_k=hybrid_k,
                final_k=final_k,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight
            )
            
            logger.info(f"Hybrid search with reranking found {len(results)} final results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search with reranking: {str(e)}")
            return []
    
    def get_hybrid_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the hybrid search engine."""
        try:
            stats = self.hybrid_engine.get_search_stats()
            stats["vector_store_info"] = self.get_collection_info()
            return stats
        except Exception as e:
            logger.error(f"Error getting hybrid search stats: {str(e)}")
            return {"error": str(e)}
