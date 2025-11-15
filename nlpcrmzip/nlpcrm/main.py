"""
Main Script for NLP CRM System - Phase 1
Data Foundation & Core RAG Implementation
"""

import logging
import argparse
from pathlib import Path
from vector_store import VectorStoreManager
from document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_knowledge_base(kb_file_path: str = "customer_service_kb.md"):
    """
    Set up the knowledge base from the customer service KB file.
    
    Args:
        kb_file_path: Path to the knowledge base file
    """
    logger.info("Setting up knowledge base...")
    
    try:
        # Initialize vector store manager
        vs_manager = VectorStoreManager()
        
        # Set up the knowledge base
        success = vs_manager.setup_knowledge_base(kb_file_path)
        
        if success:
            logger.info("✓ Knowledge base setup completed successfully!")
            
            # Display collection info
            info = vs_manager.get_collection_info()
            logger.info(f"Collection info: {info}")
            
            return vs_manager
        else:
            logger.error("✗ Knowledge base setup failed!")
            return None
            
    except Exception as e:
        logger.error(f"Error setting up knowledge base: {str(e)}")
        return None

def interactive_search(vs_manager: VectorStoreManager):
    """
    Interactive search interface for testing semantic retrieval.
    
    Args:
        vs_manager: Initialized VectorStoreManager instance
    """
    logger.info("Starting interactive search mode...")
    logger.info("Commands: 'quit' to exit, 'info' for collection info, 'stats' for hybrid search stats")
    logger.info("Search modes: 'vector', 'hybrid', 'rerank' (default: rerank)")
    
    while True:
        try:
            query = input("\nEnter your search query: ").strip()
            
            if query.lower() == 'quit':
                logger.info("Exiting interactive search mode...")
                break
            elif query.lower() == 'info':
                info = vs_manager.get_collection_info()
                print(f"Collection info: {info}")
                continue
            elif query.lower() == 'stats':
                stats = vs_manager.get_hybrid_search_stats()
                print(f"Hybrid search stats: {stats}")
                continue
            elif not query:
                print("Please enter a valid query.")
                continue
            
            # Ask for search mode
            mode = input("Search mode (vector/hybrid/rerank) [rerank]: ").strip().lower()
            if not mode:
                mode = "rerank"
            
            if mode == "vector":
                # Perform vector similarity search
                results = vs_manager.similarity_search(query, k=3)
                print(f"\nVector Search Results ({len(results)} found):")
                print("-" * 50)
                
                for i, doc in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print(f"Content: {doc.page_content}")
                    print(f"Metadata: {doc.metadata}")
                    print("-" * 30)
                    
            elif mode == "hybrid":
                # Perform hybrid search
                results = vs_manager.hybrid_search(query, k=5)
                print(f"\nHybrid Search Results ({len(results)} found):")
                print("-" * 50)
                
                for i, (doc, score) in enumerate(results, 1):
                    print(f"\nResult {i} (hybrid score: {score:.4f}):")
                    print(f"Content: {doc.page_content}")
                    print(f"Metadata: {doc.metadata}")
                    print("-" * 30)
                    
            elif mode == "rerank":
                # Perform hybrid search with reranking
                results = vs_manager.hybrid_search_with_reranking(query, hybrid_k=10, final_k=3)
                print(f"\nReranked Search Results ({len(results)} found):")
                print("-" * 50)
                
                for i, (doc, score) in enumerate(results, 1):
                    print(f"\nResult {i} (rerank score: {score:.4f}):")
                    print(f"Content: {doc.page_content}")
                    print(f"Metadata: {doc.metadata}")
                    print("-" * 30)
            else:
                print("Invalid search mode. Using rerank mode.")
                results = vs_manager.hybrid_search_with_reranking(query, hybrid_k=10, final_k=3)
                print(f"\nReranked Search Results ({len(results)} found):")
                print("-" * 50)
                
                for i, (doc, score) in enumerate(results, 1):
                    print(f"\nResult {i} (rerank score: {score:.4f}):")
                    print(f"Content: {doc.page_content}")
                    print(f"Metadata: {doc.metadata}")
                    print("-" * 30)
                
        except KeyboardInterrupt:
            logger.info("\nExiting interactive search mode...")
            break
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")

def run_tests():
    """Run the test suite."""
    logger.info("Running test suite...")
    
    try:
        from test_retrieval import run_comprehensive_test
        success = run_comprehensive_test()
        
        if success:
            logger.info("✓ All tests passed!")
        else:
            logger.error("✗ Some tests failed!")
            
        return success
        
    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        return False

def main():
    """Main function to run the NLP CRM system."""
    parser = argparse.ArgumentParser(description="NLP CRM System - Phase 1: Data Foundation & Core RAG")
    parser.add_argument(
        "--mode", 
        choices=["setup", "search", "test"], 
        default="setup",
        help="Mode to run: setup (default), search, or test"
    )
    parser.add_argument(
        "--kb-file", 
        default="customer_service_kb.md",
        help="Path to the knowledge base file"
    )
    parser.add_argument(
        "--reset", 
        action="store_true",
        help="Reset the vector store collection before setup"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("NLP CRM SYSTEM - PHASE 1 & 2: DATA FOUNDATION & RETRIEVAL ROBUSTNESS")
    logger.info("=" * 60)
    
    if args.mode == "test":
        # Run tests
        success = run_tests()
        return success
    
    # Initialize vector store manager
    vs_manager = VectorStoreManager()
    
    if args.reset:
        logger.info("Resetting vector store collection...")
        vs_manager.reset_collection()
    
    if args.mode == "setup":
        # Set up knowledge base
        vs_manager = setup_knowledge_base(args.kb_file)
        
        if vs_manager:
            logger.info("\nKnowledge base setup complete!")
            logger.info("You can now run the system in search mode with: python main.py --mode search")
        
    elif args.mode == "search":
        # Check if knowledge base exists
        info = vs_manager.get_collection_info()
        if info.get("document_count", 0) == 0:
            logger.warning("No documents found in the knowledge base.")
            logger.info("Setting up knowledge base first...")
            vs_manager = setup_knowledge_base(args.kb_file)
            
            if not vs_manager:
                logger.error("Failed to set up knowledge base. Cannot proceed with search.")
                return False
        
        # Start interactive search
        interactive_search(vs_manager)
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\nNLP CRM System Phase 1 & 2 completed successfully!")
    else:
        print("\nNLP CRM System Phase 1 & 2 encountered errors.")
