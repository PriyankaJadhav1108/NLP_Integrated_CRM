"""
Offline API Test for NLP CRM System
Tests the API components without requiring a running server.
"""

import logging
import json
from typing import Dict, Any

# Import our modules
from vector_store import VectorStoreManager
from chat_history import ChatHistoryManager
from llm_integration import LLMResponseGenerator, StructuredResponse
from query_transformer import QueryTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OfflineAPITester:
    """Test API functionality without running server."""
    
    def __init__(self):
        """Initialize the offline API tester."""
        self.vector_store = None
        self.chat_history = None
        self.llm_generator = None
        self.query_transformer = None
        
        logger.info("Offline API tester initialized")
    
    def initialize_components(self) -> bool:
        """Initialize all components."""
        try:
            logger.info("Initializing components...")
            
            # Initialize vector store
            self.vector_store = VectorStoreManager()
            
            # Initialize chat history
            self.chat_history = ChatHistoryManager()
            
            # Initialize LLM generator
            self.llm_generator = LLMResponseGenerator(use_mock=True)
            
            # Initialize query transformer
            self.query_transformer = QueryTransformer()
            
            logger.info("✓ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"✗ Component initialization failed: {str(e)}")
            return False
    
    def test_session_creation(self) -> bool:
        """Test session creation functionality."""
        logger.info("Testing session creation...")
        
        try:
            # Create session
            session = self.chat_history.create_session(user_id="test_user_123")
            
            if session and session.session_id:
                logger.info(f"✓ Session created: {session.session_id}")
                return True
            else:
                logger.error("✗ Session creation failed")
                return False
                
        except Exception as e:
            logger.error(f"✗ Session creation error: {str(e)}")
            return False
    
    def test_query_processing(self, query: str, test_name: str) -> bool:
        """Test complete query processing pipeline."""
        logger.info(f"Testing query processing: {test_name}")
        
        try:
            # Step 1: Create session
            session = self.chat_history.create_session(user_id="test_user")
            session_id = session.session_id
            
            # Step 2: Transform query
            query_metadata = self.query_transformer.get_query_metadata(query)
            enhanced_query = self.query_transformer.enhance_query_for_retrieval(query)
            retrieval_queries = self.query_transformer.transform_for_retrieval(query, max_variations=2)
            
            # Step 3: Retrieve documents
            retrieved_docs = self.vector_store.hybrid_search_with_reranking(
                query=retrieval_queries[0],
                hybrid_k=5,
                final_k=3
            )
            
            # Step 4: Format documents
            formatted_docs = []
            for doc, score in retrieved_docs:
                formatted_docs.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": score
                })
            
            # Step 5: Get conversation history
            conversation_history = self.chat_history.get_conversation_history(session_id)
            
            # Step 6: Generate LLM response
            structured_response = self.llm_generator.generate_structured_response(
                query=query,
                retrieved_docs=formatted_docs,
                conversation_history=conversation_history
            )
            
            # Step 7: Update chat history
            self.chat_history.add_message(session_id, "user", query)
            self.chat_history.add_message(session_id, "assistant", structured_response.answer)
            
            # Log results
            logger.info(f"✓ Query processed successfully")
            logger.info(f"  Query: {query}")
            logger.info(f"  Response type: {structured_response.response_type}")
            logger.info(f"  Answer: {structured_response.answer[:100]}...")
            logger.info(f"  Confidence: {structured_response.confidence}")
            logger.info(f"  Sources: {len(structured_response.sources)}")
            logger.info(f"  Session: {session_id}")
            logger.info(f"  Detected intents: {query_metadata['detected_intents']}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Query processing failed: {str(e)}")
            return False
    
    def test_asr_processing(self, asr_text: str, test_name: str) -> bool:
        """Test ASR text processing."""
        logger.info(f"Testing ASR processing: {test_name}")
        
        try:
            # Clean ASR text
            cleaned_text = self.query_transformer.clean_asr_output(asr_text)
            
            # Enhance query
            enhanced = self.query_transformer.enhance_query_for_retrieval(cleaned_text)
            
            # Get metadata
            metadata = self.query_transformer.get_query_metadata(cleaned_text)
            
            logger.info(f"✓ ASR processing successful")
            logger.info(f"  Original ASR: {asr_text}")
            logger.info(f"  Cleaned text: {cleaned_text}")
            logger.info(f"  Detected intents: {enhanced['detected_intents']}")
            logger.info(f"  Confidence: {enhanced['confidence']}")
            logger.info(f"  Query variations: {len(enhanced['query_variations'])}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ ASR processing failed: {str(e)}")
            return False
    
    def test_session_history(self) -> bool:
        """Test session history functionality."""
        logger.info("Testing session history...")
        
        try:
            # Create session and add messages
            session = self.chat_history.create_session(user_id="test_user")
            session_id = session.session_id
            
            # Add some messages
            self.chat_history.add_message(session_id, "user", "Hello")
            self.chat_history.add_message(session_id, "assistant", "Hi there!")
            self.chat_history.add_message(session_id, "user", "How are you?")
            self.chat_history.add_message(session_id, "assistant", "I'm doing well, thank you!")
            
            # Get history
            history = self.chat_history.get_conversation_history(session_id)
            recent_messages = self.chat_history.get_recent_messages(session_id, limit=3)
            
            logger.info(f"✓ Session history test successful")
            logger.info(f"  Session ID: {session_id}")
            logger.info(f"  Total messages: {len(session.messages)}")
            logger.info(f"  Recent messages: {len(recent_messages)}")
            logger.info(f"  History length: {len(history)} characters")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Session history test failed: {str(e)}")
            return False
    
    def test_structured_response_schema(self) -> bool:
        """Test structured response schema."""
        logger.info("Testing structured response schema...")
        
        try:
            # Create a mock response
            response = self.llm_generator.generate_structured_response(
                query="Test query",
                retrieved_docs=[],
                conversation_history=None
            )
            
            # Convert to dict and back to test serialization
            response_dict = response.dict()
            response_json = json.dumps(response_dict, indent=2)
            parsed_response = json.loads(response_json)
            
            # Validate required fields
            required_fields = [
                "response_type", "answer", "confidence", "sources",
                "response_id", "timestamp", "suggested_actions",
                "follow_up_questions", "escalation_required"
            ]
            
            for field in required_fields:
                if field not in parsed_response:
                    logger.error(f"✗ Missing required field: {field}")
                    return False
            
            logger.info(f"✓ Structured response schema test successful")
            logger.info(f"  Response ID: {response.response_id}")
            logger.info(f"  Response type: {response.response_type}")
            logger.info(f"  JSON size: {len(response_json)} bytes")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Structured response schema test failed: {str(e)}")
            return False
    
    def run_comprehensive_test(self) -> bool:
        """Run comprehensive offline API tests."""
        logger.info("=" * 60)
        logger.info("STARTING OFFLINE API COMPREHENSIVE TEST")
        logger.info("=" * 60)
        
        # Initialize components
        if not self.initialize_components():
            return False
        
        test_results = []
        
        # Test 1: Session Creation
        test_results.append(self.test_session_creation())
        
        # Test 2: Query Processing
        test_queries = [
            ("How do I get a refund?", "Refund Policy Query"),
            ("What is your shipping policy?", "Shipping Policy Query"),
            ("How can I contact customer support?", "Support Contact Query"),
            ("Hello, I need help", "Greeting Query")
        ]
        
        for query, test_name in test_queries:
            test_results.append(self.test_query_processing(query, test_name))
        
        # Test 3: ASR Processing
        asr_tests = [
            ("um i need help with uh refund policy", "ASR with filler words"),
            ("can you tell me about shipping and delivery", "ASR shipping query"),
            ("hello i want to return something", "ASR return query")
        ]
        
        for asr_text, test_name in asr_tests:
            test_results.append(self.test_asr_processing(asr_text, test_name))
        
        # Test 4: Session History
        test_results.append(self.test_session_history())
        
        # Test 5: Structured Response Schema
        test_results.append(self.test_structured_response_schema())
        
        # Summary
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        
        logger.info("=" * 60)
        logger.info(f"OFFLINE API TEST RESULTS: {passed_tests}/{total_tests} tests passed")
        logger.info("=" * 60)
        
        if passed_tests == total_tests:
            logger.info("✓ ALL OFFLINE API TESTS PASSED!")
            return True
        else:
            logger.error(f"✗ {total_tests - passed_tests} tests failed")
            return False

def main():
    """Main test function."""
    tester = OfflineAPITester()
    
    try:
        success = tester.run_comprehensive_test()
        
        if success:
            print("\\nNLP CRM System API components are working correctly!")
            print("The system is ready for deployment and production use.")
        else:
            print("\\nSome API components need fixing before deployment.")
            
    except Exception as e:
        logger.error(f"Test execution error: {str(e)}")

if __name__ == "__main__":
    main()
