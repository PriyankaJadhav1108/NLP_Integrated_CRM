"""
Test Script for Semantic Retrieval Functionality
Tests the document processing and vector store operations.
"""

import logging
from typing import List
from vector_store import VectorStoreManager
from document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_document_processing():
    """Test document processing functionality."""
    logger.info("Testing document processing...")
    
    try:
        processor = DocumentProcessor()
        documents = processor.process_customer_service_kb()
        
        logger.info(f"✓ Document processing successful: {len(documents)} chunks created")
        
        # Print sample chunk
        if documents:
            sample_chunk = documents[0]
            logger.info(f"Sample chunk content: {sample_chunk.page_content[:200]}...")
            logger.info(f"Sample chunk metadata: {sample_chunk.metadata}")
        
        return documents
        
    except Exception as e:
        logger.error(f"✗ Document processing failed: {str(e)}")
        return None

def test_vector_store_operations():
    """Test vector store operations."""
    logger.info("Testing vector store operations...")
    
    try:
        # Initialize vector store manager
        vs_manager = VectorStoreManager()
        
        # Get collection info
        info = vs_manager.get_collection_info()
        logger.info(f"✓ Vector store initialized: {info}")
        
        return vs_manager
        
    except Exception as e:
        logger.error(f"✗ Vector store initialization failed: {str(e)}")
        return None

def test_knowledge_base_setup(vs_manager: VectorStoreManager):
    """Test knowledge base setup."""
    logger.info("Testing knowledge base setup...")
    
    try:
        success = vs_manager.setup_knowledge_base()
        
        if success:
            logger.info("✓ Knowledge base setup successful")
            
            # Get updated collection info
            info = vs_manager.get_collection_info()
            logger.info(f"Collection info after setup: {info}")
            
            return True
        else:
            logger.error("✗ Knowledge base setup failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ Knowledge base setup error: {str(e)}")
        return False

def test_semantic_retrieval(vs_manager: VectorStoreManager):
    """Test semantic retrieval functionality."""
    logger.info("Testing semantic retrieval...")
    
    # Test queries
    test_queries = [
        "How do I get a refund?",
        "What is the shipping policy?",
        "How can I contact customer support?",
        "What are the warranty terms?",
        "How do I reset my password?",
        "What payment methods do you accept?"
    ]
    
    try:
        for query in test_queries:
            logger.info(f"\n--- Testing query: '{query}' ---")
            
            # Test basic similarity search
            results = vs_manager.similarity_search(query, k=3)
            
            if results:
                logger.info(f"✓ Found {len(results)} relevant documents")
                for i, doc in enumerate(results):
                    logger.info(f"  Result {i+1}: {doc.page_content[:150]}...")
                    logger.info(f"  Metadata: {doc.metadata}")
            else:
                logger.warning(f"✗ No results found for query: {query}")
            
            # Test similarity search with scores
            results_with_scores = vs_manager.similarity_search_with_score(query, k=3)
            
            if results_with_scores:
                logger.info(f"✓ Found {len(results_with_scores)} results with scores")
                for i, (doc, score) in enumerate(results_with_scores):
                    logger.info(f"  Result {i+1} (score: {score:.4f}): {doc.page_content[:100]}...")
        
        logger.info("✓ Semantic retrieval testing completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Semantic retrieval testing failed: {str(e)}")
        return False

def test_hybrid_search(vs_manager: VectorStoreManager):
    """Test hybrid search functionality."""
    logger.info("Testing hybrid search...")
    
    # Test queries
    test_queries = [
        "refund policy 30 days",
        "free shipping orders",
        "customer support email phone",
        "warranty information products",
        "password reset login",
        "payment methods credit card"
    ]
    
    try:
        for query in test_queries:
            logger.info(f"\n--- Testing hybrid search query: '{query}' ---")
            
            # Test hybrid search
            hybrid_results = vs_manager.hybrid_search(query, k=5)
            
            if hybrid_results:
                logger.info(f"✓ Hybrid search found {len(hybrid_results)} results")
                for i, (doc, score) in enumerate(hybrid_results):
                    logger.info(f"  Result {i+1} (hybrid score: {score:.4f}): {doc.page_content[:100]}...")
            else:
                logger.warning(f"✗ No hybrid results found for query: {query}")
        
        logger.info("✓ Hybrid search testing completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Hybrid search testing failed: {str(e)}")
        return False

def test_reranking(vs_manager: VectorStoreManager):
    """Test reranking functionality."""
    logger.info("Testing reranking...")
    
    # Test queries
    test_queries = [
        "How do I get a refund?",
        "What is the shipping policy?",
        "How can I contact customer support?"
    ]
    
    try:
        for query in test_queries:
            logger.info(f"\n--- Testing reranking query: '{query}' ---")
            
            # Test hybrid search with reranking
            reranked_results = vs_manager.hybrid_search_with_reranking(
                query=query,
                hybrid_k=10,
                final_k=3
            )
            
            if reranked_results:
                logger.info(f"✓ Reranking found {len(reranked_results)} final results")
                for i, (doc, score) in enumerate(reranked_results):
                    logger.info(f"  Result {i+1} (rerank score: {score:.4f}): {doc.page_content[:100]}...")
            else:
                logger.warning(f"✗ No reranked results found for query: {query}")
        
        logger.info("✓ Reranking testing completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Reranking testing failed: {str(e)}")
        return False

def run_comprehensive_test():
    """Run comprehensive test of the entire system."""
    logger.info("=" * 60)
    logger.info("STARTING COMPREHENSIVE NLP CRM SYSTEM TEST")
    logger.info("=" * 60)
    
    # Test 1: Document Processing
    documents = test_document_processing()
    if not documents:
        logger.error("Document processing test failed. Stopping tests.")
        return False
    
    # Test 2: Vector Store Operations
    vs_manager = test_vector_store_operations()
    if not vs_manager:
        logger.error("Vector store test failed. Stopping tests.")
        return False
    
    # Test 3: Knowledge Base Setup
    kb_success = test_knowledge_base_setup(vs_manager)
    if not kb_success:
        logger.error("Knowledge base setup test failed. Stopping tests.")
        return False
    
    # Test 4: Semantic Retrieval
    retrieval_success = test_semantic_retrieval(vs_manager)
    if not retrieval_success:
        logger.error("Semantic retrieval test failed.")
        return False
    
    # Test 5: Hybrid Search
    hybrid_success = test_hybrid_search(vs_manager)
    if not hybrid_success:
        logger.error("Hybrid search test failed.")
        return False
    
    # Test 6: Reranking
    reranking_success = test_reranking(vs_manager)
    if not reranking_success:
        logger.error("Reranking test failed.")
        return False
    
    logger.info("=" * 60)
    logger.info("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
    logger.info("=" * 60)
    
    return True

if __name__ == "__main__":
    success = run_comprehensive_test()
    
    if success:
        print("\nAll tests passed! Your NLP CRM system is ready for use.")
    else:
        print("\nSome tests failed. Please check the logs for details.")
