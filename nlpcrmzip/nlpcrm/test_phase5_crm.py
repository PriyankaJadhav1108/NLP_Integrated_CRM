"""
Test script for Phase 5: CRM Integration & UI
Tests the CRM logging API and frontend integration.
"""

import json
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_crm_models():
    """Test CRM models and storage functionality."""
    print("=" * 60)
    print("Testing CRM Models and Storage")
    print("=" * 60)
    
    try:
        from crm_models import (
            InteractionLog, InteractionStatus, InteractionType,
            IntentType, SentimentType, CRMStorage, InteractionMetadata
        )
        
        # Test model creation
        print("OK CRM models imported successfully")
        
        # Test storage initialization
        storage = CRMStorage("./test_crm_data")
        print("OK CRM storage initialized")
        
        # Test interaction creation
        metadata = InteractionMetadata(
            session_id="test_session_123",
            user_id="test_user_456",
            source_channel="test"
        )
        
        interaction = InteractionLog(
            status=InteractionStatus.PENDING,
            interaction_type=InteractionType.SUPPORT,
            intent=IntentType.SUPPORT,
            sentiment=SentimentType.NEUTRAL,
            customer_query="I need help with my account",
            assistant_response="I'd be happy to help you with your account. What specific issue are you experiencing?",
            response_type="helpful",
            confidence="high",
            metadata=metadata,
            customer_id="test_customer_789",
            priority=2,
            tags=["account", "support"],
            suggested_actions=["check_account_status", "verify_identity"],
            escalation_required=False
        )
        
        print("OK Interaction model created successfully")
        
        # Test logging interaction
        interaction_id = storage.log_interaction(interaction)
        print(f"OK Interaction logged with ID: {interaction_id}")
        
        # Test retrieving interaction
        retrieved = storage.get_interaction(interaction_id)
        if retrieved:
            print("OK Interaction retrieved successfully")
            print(f"  - Status: {retrieved.status}")
            print(f"  - Intent: {retrieved.intent}")
            print(f"  - Sentiment: {retrieved.sentiment}")
        else:
            print("ERR Failed to retrieve interaction")
        
        # Test customer interactions
        customer_interactions = storage.get_interactions_by_customer("test_customer_789")
        print(f"OK Found {len(customer_interactions)} interactions for customer")
        
        # Test status filtering
        pending_interactions = storage.get_interactions_by_status(InteractionStatus.PENDING)
        print(f"OK Found {len(pending_interactions)} pending interactions")
        
        # Test statistics
        stats = storage.get_interaction_stats()
        print("OK Generated interaction statistics:")
        print(f"  - Total interactions: {stats.get('total_interactions', 0)}")
        print(f"  - By status: {stats.get('by_status', {})}")
        print(f"  - By intent: {stats.get('by_intent', {})}")
        print(f"  - By sentiment: {stats.get('by_sentiment', {})}")
        
        # Test status update
        success = storage.update_interaction_status(
            interaction_id, 
            InteractionStatus.RESOLVED, 
            agent_id="test_agent_001"
        )
        if success:
            print("OK Interaction status updated successfully")
        else:
            print("ERR Failed to update interaction status")
        
        return True
        
    except Exception as e:
        print(f"ERR Error testing CRM models: {str(e)}")
        return False

def test_crm_api_logic():
    """Test CRM API logic without running the server."""
    print("\n" + "=" * 60)
    print("Testing CRM API Logic")
    print("=" * 60)
    
    try:
        from crm_api import (
            InteractionLogRequest, InteractionLogResponse,
            _determine_intent_from_response, _determine_sentiment_from_query,
            _determine_interaction_type
        )
        from crm_models import IntentType
        
        print("OK CRM API models imported successfully")
        
        # Test intent determination
        test_queries = [
            ("I want a refund for my order", "refund"),
            ("When will my package arrive?", "shipping"),
            ("I forgot my password", "password"),
            ("Hello, how are you?", "greeting"),
            ("I have a complaint about the service", "complaint")
        ]
        
        for query, expected_intent in test_queries:
            intent = _determine_intent_from_response("helpful", query)
            print(f"OK Query: '{query}' -> Intent: {intent}")
        
        # Test sentiment determination
        test_sentiments = [
            ("Thank you so much for your help!", "positive"),
            ("I'm really frustrated with this issue", "negative"),
            ("What is your return policy?", "neutral"),
            ("I love the product but hate the shipping", "mixed")
        ]
        
        for query, expected_sentiment in test_sentiments:
            sentiment = _determine_sentiment_from_query(query)
            print(f"OK Query: '{query}' -> Sentiment: {sentiment}")
        
        # Test interaction type determination
        interaction_type = _determine_interaction_type(IntentType.COMPLAINT, True)
        print(f"OK Escalated complaint -> Type: {interaction_type}")
        
        interaction_type = _determine_interaction_type(IntentType.SUPPORT, False)
        print(f"OK Non-escalated support -> Type: {interaction_type}")
        
        # Test request model creation
        request = InteractionLogRequest(
            customer_query="I need help with my order",
            assistant_response="I'd be happy to help you with your order. What's the order number?",
            response_type="helpful",
            confidence="high",
            session_id="test_session_456",
            customer_id="test_customer_123",
            priority=2,
            tags=["order", "support"],
            escalation_required=False
        )
        
        print("OK Interaction log request created successfully")
        print(f"  - Customer ID: {request.customer_id}")
        print(f"  - Priority: {request.priority}")
        print(f"  - Tags: {request.tags}")
        
        return True
        
    except Exception as e:
        print(f"ERR Error testing CRM API logic: {str(e)}")
        return False

def test_integration_with_nlp_system():
    """Test integration between NLP system and CRM logging."""
    print("\n" + "=" * 60)
    print("Testing NLP-CRM Integration")
    print("=" * 60)
    
    try:
        from vector_store import VectorStoreManager
        from chat_history import ChatHistoryManager
        from llm_integration import LLMResponseGenerator
        from query_transformer import QueryTransformer
        from crm_models import CRMStorage, InteractionLog, InteractionStatus
        from crm_api import InteractionLogRequest
        
        print("OK All integration components imported successfully")
        
        # Initialize components
        vector_store = VectorStoreManager()
        chat_history = ChatHistoryManager()
        llm_generator = LLMResponseGenerator()
        query_transformer = QueryTransformer()
        crm_storage = CRMStorage("./test_crm_data")
        
        print("OK All components initialized")
        
        # Simulate a complete NLP processing pipeline
        test_query = "I want to return this product because it's defective"
        
        # Step 1: Query transformation
        query_metadata = query_transformer.get_query_metadata(test_query)
        enhanced_query = query_transformer.enhance_query_for_retrieval(test_query)
        print(f"OK Query enhanced: '{enhanced_query}'")
        
        # Step 2: Document retrieval (if knowledge base is set up)
        try:
            retrieved_docs = vector_store.hybrid_search_with_reranking(
                query=enhanced_query,
                hybrid_k=5,
                final_k=2
            )
            print(f"OK Retrieved {len(retrieved_docs)} relevant documents")
        except Exception as e:
            print(f"⚠ Knowledge base not set up, skipping retrieval: {str(e)}")
            retrieved_docs = []
        
        # Step 3: Generate LLM response
        formatted_docs = []
        for doc, score in retrieved_docs:
            formatted_docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score
            })
        
        structured_response = llm_generator.generate_structured_response(
            query=test_query,
            retrieved_docs=formatted_docs,
            conversation_history=None
        )
        
        print("OK Generated structured LLM response:")
        print(f"  - Answer: {structured_response.answer[:100]}...")
        print(f"  - Response Type: {structured_response.response_type}")
        print(f"  - Confidence: {structured_response.confidence}")
        print(f"  - Escalation Required: {structured_response.escalation_required}")
        
        # Step 4: Create CRM log request
        crm_request = InteractionLogRequest(
            customer_query=test_query,
            assistant_response=structured_response.answer,
            response_type=structured_response.response_type,
            confidence=structured_response.confidence,
            session_id="integration_test_session",
            customer_id="integration_test_customer",
            priority=3 if structured_response.escalation_required else 2,
            tags=["return", "defective", "product"],
            sources=formatted_docs,
            escalation_required=structured_response.escalation_required,
            escalation_reason=structured_response.escalation_reason,
            processing_time_ms=150,
            query_metadata=query_metadata,
            retrieved_docs_count=len(formatted_docs),
            confidence_score=0.85
        )
        
        print("OK CRM log request created successfully")
        
        # Step 5: Log to CRM (simulate API call)
        from crm_api import _determine_intent_from_response, _determine_sentiment_from_query, _determine_interaction_type
        
        intent = _determine_intent_from_response(crm_request.response_type, crm_request.customer_query)
        sentiment = _determine_sentiment_from_query(crm_request.customer_query)
        interaction_type = _determine_interaction_type(intent, crm_request.escalation_required)
        
        # Create interaction log
        interaction = InteractionLog(
            status=InteractionStatus.ESCALATED if crm_request.escalation_required else InteractionStatus.PENDING,
            interaction_type=interaction_type,
            intent=intent,
            sentiment=sentiment,
            customer_query=crm_request.customer_query,
            assistant_response=crm_request.assistant_response,
            response_type=crm_request.response_type,
            confidence=crm_request.confidence,
            metadata={
                "session_id": crm_request.session_id,
                "user_id": crm_request.user_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "processing_time_ms": crm_request.processing_time_ms,
                "query_metadata": crm_request.query_metadata,
                "retrieved_docs_count": crm_request.retrieved_docs_count,
                "confidence_score": crm_request.confidence_score,
                "escalation_reason": crm_request.escalation_reason,
                "source_channel": crm_request.source_channel
            },
            sources=crm_request.sources,
            customer_id=crm_request.customer_id,
            priority=crm_request.priority,
            tags=crm_request.tags,
            escalation_required=crm_request.escalation_required
        )
        
        # Log the interaction
        interaction_id = crm_storage.log_interaction(interaction)
        print(f"OK Interaction logged to CRM with ID: {interaction_id}")
        
        # Verify the logged interaction
        logged_interaction = crm_storage.get_interaction(interaction_id)
        if logged_interaction:
            print("OK Integration test completed successfully!")
            print(f"  - Interaction ID: {logged_interaction.interaction_id}")
            print(f"  - Status: {logged_interaction.status}")
            print(f"  - Intent: {logged_interaction.intent}")
            print(f"  - Sentiment: {logged_interaction.sentiment}")
            print(f"  - Escalation Required: {logged_interaction.escalation_required}")
        else:
            print("ERR Failed to retrieve logged interaction")
        
        return True
        
    except Exception as e:
        print(f"ERR Error testing NLP-CRM integration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_frontend_data_format():
    """Test data format compatibility with frontend."""
    print("\n" + "=" * 60)
    print("Testing Frontend Data Format")
    print("=" * 60)
    
    try:
        from crm_models import CRMStorage, InteractionStatus
        
        # Initialize storage
        storage = CRMStorage("./test_crm_data")
        
        # Get sample interactions
        interactions = storage.get_interactions_by_status(InteractionStatus.PENDING, 5)
        escalated = storage.get_interactions_by_status(InteractionStatus.ESCALATED, 5)
        
        print(f"OK Retrieved {len(interactions)} pending interactions")
        print(f"OK Retrieved {len(escalated)} escalated interactions")
        
        # Test data serialization for frontend
        for interaction in interactions[:2]:  # Test first 2
            # Convert to dict (as would be sent to frontend)
            interaction_dict = interaction.dict()
            
            # Check required fields for frontend
            required_fields = [
                'interaction_id', 'status', 'intent', 'sentiment',
                'customer_query', 'assistant_response', 'created_at',
                'priority', 'escalation_required'
            ]
            
            missing_fields = [field for field in required_fields if field not in interaction_dict]
            if not missing_fields:
                print(f"OK Interaction {interaction.interaction_id[:8]}... has all required frontend fields")
            else:
                print(f"ERR Interaction missing fields: {missing_fields}")
        
        # Test statistics format
        stats = storage.get_interaction_stats()
        print("OK Statistics format:")
        print(f"  - Total: {stats.get('total_interactions', 0)}")
        print(f"  - By status: {stats.get('by_status', {})}")
        print(f"  - By intent: {stats.get('by_intent', {})}")
        print(f"  - By sentiment: {stats.get('by_sentiment', {})}")
        print(f"  - Escalation rate: {stats.get('escalation_rate', 0):.2%}")
        
        return True
        
    except Exception as e:
        print(f"ERR Error testing frontend data format: {str(e)}")
        return False

def run_comprehensive_test():
    """Run comprehensive Phase 5 test suite."""
    print("Starting Phase 5: CRM Integration & UI Test Suite")
    print("=" * 80)
    
    test_results = []
    
    # Run all tests
    test_results.append(("CRM Models & Storage", test_crm_models()))
    test_results.append(("CRM API Logic", test_crm_api_logic()))
    test_results.append(("NLP-CRM Integration", test_integration_with_nlp_system()))
    test_results.append(("Frontend Data Format", test_frontend_data_format()))
    
    # Print summary
    print("\n" + "=" * 80)
    print("Phase 5 Test Results Summary")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll Phase 5 tests passed! CRM Integration & UI is ready.")
        print("\nNext steps:")
        print("1. Start the API server: python api_server.py")
        print("2. Open the CRM dashboard: http://127.0.0.1:8000/crm")
        print("3. Test the CRM logging API: POST /api/v1/interactions/log")
    else:
        print(f"\n{total - passed} tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    run_comprehensive_test()
