"""
Test Phase 4 Components Individually
Tests each component before full API integration.
"""

import logging
import asyncio
from typing import Dict, Any

# Import our modules
from vector_store import VectorStoreManager
from chat_history import ChatHistoryManager
from llm_integration import LLMResponseGenerator, StructuredResponse
from query_transformer import QueryTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vector_store():
    """Test vector store functionality."""
    logger.info("Testing vector store...")
    
    try:
        vs_manager = VectorStoreManager()
        
        # Test basic search
        results = vs_manager.similarity_search("refund policy", k=3)
        logger.info(f"✓ Vector store search: {len(results)} results")
        
        # Test hybrid search
        hybrid_results = vs_manager.hybrid_search("shipping information", k=3)
        logger.info(f"✓ Hybrid search: {len(hybrid_results)} results")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Vector store test failed: {str(e)}")
        return False

def test_chat_history():
    """Test chat history functionality."""
    logger.info("Testing chat history...")
    
    try:
        chat_manager = ChatHistoryManager()
        
        # Create session
        session = chat_manager.create_session(user_id="test_user")
        logger.info(f"✓ Session created: {session.session_id}")
        
        # Add messages
        chat_manager.add_message(session.session_id, "user", "Hello, I need help")
        chat_manager.add_message(session.session_id, "assistant", "I'm here to help!")
        
        # Get history
        history = chat_manager.get_conversation_history(session.session_id)
        logger.info(f"✓ Chat history retrieved: {len(history.split('\\n'))} messages")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Chat history test failed: {str(e)}")
        return False

def test_llm_integration():
    """Test LLM integration."""
    logger.info("Testing LLM integration...")
    
    try:
        llm_generator = LLMResponseGenerator(use_mock=True)
        
        # Mock retrieved documents
        mock_docs = [
            {
                "page_content": "Our refund policy allows returns within 30 days.",
                "metadata": {"source": "test.md"},
                "relevance_score": 0.9
            }
        ]
        
        # Generate response
        response = llm_generator.generate_structured_response(
            query="How do I get a refund?",
            retrieved_docs=mock_docs,
            conversation_history=None
        )
        
        logger.info(f"✓ LLM response generated: {response.response_type}")
        logger.info(f"  Answer: {response.answer[:50]}...")
        logger.info(f"  Confidence: {response.confidence}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ LLM integration test failed: {str(e)}")
        return False

def test_query_transformer():
    """Test query transformation."""
    logger.info("Testing query transformer...")
    
    try:
        transformer = QueryTransformer()
        
        # Test query enhancement
        enhanced = transformer.enhance_query_for_retrieval("um i need help with uh refund")
        logger.info(f"✓ Query enhanced: {enhanced['cleaned_query']}")
        logger.info(f"  Intents: {enhanced['detected_intents']}")
        logger.info(f"  Confidence: {enhanced['confidence']}")
        
        # Test query variations
        variations = transformer.transform_for_retrieval("refund policy", max_variations=3)
        logger.info(f"✓ Query variations: {len(variations)} generated")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Query transformer test failed: {str(e)}")
        return False

def test_integrated_pipeline():
    """Test the integrated pipeline."""
    logger.info("Testing integrated pipeline...")
    
    try:
        # Initialize components
        vs_manager = VectorStoreManager()
        chat_manager = ChatHistoryManager()
        llm_generator = LLMResponseGenerator(use_mock=True)
        transformer = QueryTransformer()
        
        # Create session
        session = chat_manager.create_session(user_id="test_user")
        
        # Process query
        query = "How do I get a refund?"
        
        # Transform query
        enhanced = transformer.enhance_query_for_retrieval(query)
        retrieval_queries = transformer.transform_for_retrieval(query, max_variations=2)
        
        # Retrieve documents
        retrieved_docs = vs_manager.hybrid_search_with_reranking(
            query=retrieval_queries[0],
            hybrid_k=5,
            final_k=3
        )
        
        # Format documents
        formatted_docs = []
        for doc, score in retrieved_docs:
            formatted_docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score
            })
        
        # Generate LLM response
        response = llm_generator.generate_structured_response(
            query=query,
            retrieved_docs=formatted_docs,
            conversation_history=None
        )
        
        # Update chat history
        chat_manager.add_message(session.session_id, "user", query)
        chat_manager.add_message(session.session_id, "assistant", response.answer)
        
        logger.info(f"✓ Integrated pipeline completed successfully")
        logger.info(f"  Query: {query}")
        logger.info(f"  Response type: {response.response_type}")
        logger.info(f"  Answer: {response.answer[:100]}...")
        logger.info(f"  Sources: {len(response.sources)}")
        logger.info(f"  Session: {session.session_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Integrated pipeline test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all component tests."""
    logger.info("=" * 60)
    logger.info("TESTING PHASE 4 COMPONENTS")
    logger.info("=" * 60)
    
    tests = [
        ("Vector Store", test_vector_store),
        ("Chat History", test_chat_history),
        ("LLM Integration", test_llm_integration),
        ("Query Transformer", test_query_transformer),
        ("Integrated Pipeline", test_integrated_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\\n--- Testing {test_name} ---")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"✗ {test_name} test crashed: {str(e)}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info("\\n" + "=" * 60)
    logger.info(f"PHASE 4 COMPONENT TEST RESULTS: {passed}/{total} passed")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("✓ ALL PHASE 4 COMPONENTS WORKING!")
        return True
    else:
        logger.error(f"✗ {total - passed} components failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\\nPhase 4 components are ready for API integration!")
    else:
        print("\\nSome Phase 4 components need fixing before API integration.")
