"""
Test Script for NLP CRM System API
Tests the complete API pipeline including all Phase 4 components.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APITester:
    """Test the NLP CRM System API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API tester.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.session_id = None
        
        logger.info(f"API tester initialized with base URL: {base_url}")
    
    async def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        logger.info("Testing health check endpoint...")
        
        try:
            response = await self.client.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✓ Health check passed: {health_data['status']}")
                logger.info(f"Components status: {health_data['components']}")
                return True
            else:
                logger.error(f"✗ Health check failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Health check error: {str(e)}")
            return False
    
    async def test_create_session(self) -> bool:
        """Test session creation."""
        logger.info("Testing session creation...")
        
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/session",
                json={"user_id": "test_user_123"}
            )
            
            if response.status_code == 200:
                session_data = response.json()
                if session_data["success"]:
                    self.session_id = session_data["session_id"]
                    logger.info(f"✓ Session created successfully: {self.session_id}")
                    return True
                else:
                    logger.error(f"✗ Session creation failed: {session_data.get('error_message')}")
                    return False
            else:
                logger.error(f"✗ Session creation failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Session creation error: {str(e)}")
            return False
    
    async def test_query_endpoint(self, query: str, test_name: str) -> bool:
        """Test the main query endpoint."""
        logger.info(f"Testing query endpoint: {test_name}")
        
        try:
            request_data = {
                "query": query,
                "session_id": self.session_id,
                "include_history": True,
                "max_history_messages": 5
            }
            
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/v1/answer",
                json=request_data
            )
            end_time = time.time()
            
            if response.status_code == 200:
                response_data = response.json()
                
                if response_data["success"]:
                    structured_response = response_data["response"]
                    
                    logger.info(f"✓ Query '{query}' processed successfully")
                    logger.info(f"  Response type: {structured_response['response_type']}")
                    logger.info(f"  Answer: {structured_response['answer'][:100]}...")
                    logger.info(f"  Confidence: {structured_response['confidence']}")
                    logger.info(f"  Sources: {len(structured_response['sources'])}")
                    logger.info(f"  Processing time: {response_data['processing_time_ms']}ms")
                    logger.info(f"  Total time: {(end_time - start_time) * 1000:.0f}ms")
                    
                    return True
                else:
                    logger.error(f"✗ Query failed: {response_data.get('error_message')}")
                    return False
            else:
                logger.error(f"✗ Query failed with status: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Query error: {str(e)}")
            return False
    
    async def test_asr_query(self, asr_text: str, test_name: str) -> bool:
        """Test query with ASR text."""
        logger.info(f"Testing ASR query: {test_name}")
        
        try:
            request_data = {
                "query": "Process this ASR text",
                "asr_text": asr_text,
                "session_id": self.session_id,
                "include_history": True
            }
            
            response = await self.client.post(
                f"{self.base_url}/v1/answer",
                json=request_data
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                if response_data["success"]:
                    logger.info(f"✓ ASR query processed successfully")
                    logger.info(f"  Original ASR: {asr_text}")
                    logger.info(f"  Processed query: {response_data['query_metadata']['cleaned_query']}")
                    logger.info(f"  Detected intents: {response_data['query_metadata']['detected_intents']}")
                    return True
                else:
                    logger.error(f"✗ ASR query failed: {response_data.get('error_message')}")
                    return False
            else:
                logger.error(f"✗ ASR query failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"✗ ASR query error: {str(e)}")
            return False
    
    async def test_session_history(self) -> bool:
        """Test session history retrieval."""
        logger.info("Testing session history...")
        
        try:
            response = await self.client.get(
                f"{self.base_url}/v1/session/{self.session_id}/history"
            )
            
            if response.status_code == 200:
                history_data = response.json()
                
                if history_data["success"]:
                    messages = history_data["messages"]
                    logger.info(f"✓ Session history retrieved: {len(messages)} messages")
                    
                    for i, msg in enumerate(messages[-3:], 1):  # Show last 3 messages
                        logger.info(f"  Message {i}: {msg['role']} - {msg['content'][:50]}...")
                    
                    return True
                else:
                    logger.error("✗ Session history retrieval failed")
                    return False
            else:
                logger.error(f"✗ Session history failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Session history error: {str(e)}")
            return False
    
    async def test_system_stats(self) -> bool:
        """Test system statistics endpoint."""
        logger.info("Testing system stats...")
        
        try:
            response = await self.client.get(f"{self.base_url}/v1/stats")
            
            if response.status_code == 200:
                stats_data = response.json()
                
                if stats_data["success"]:
                    stats = stats_data["stats"]
                    logger.info("✓ System stats retrieved:")
                    logger.info(f"  Vector store documents: {stats['vector_store']['document_count']}")
                    logger.info(f"  Chat sessions: {stats['chat_history']['total_sessions']}")
                    logger.info(f"  Active sessions: {stats['chat_history']['active_sessions']}")
                    return True
                else:
                    logger.error("✗ System stats retrieval failed")
                    return False
            else:
                logger.error(f"✗ System stats failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"✗ System stats error: {str(e)}")
            return False
    
    async def run_comprehensive_test(self) -> bool:
        """Run comprehensive API tests."""
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE API TEST")
        logger.info("=" * 60)
        
        test_results = []
        
        # Test 1: Health Check
        test_results.append(await self.test_health_check())
        
        # Test 2: Create Session
        test_results.append(await self.test_create_session())
        
        if not self.session_id:
            logger.error("Cannot continue tests without session ID")
            return False
        
        # Test 3: Basic Queries
        test_queries = [
            ("How do I get a refund?", "Refund Policy Query"),
            ("What is your shipping policy?", "Shipping Policy Query"),
            ("How can I contact customer support?", "Support Contact Query"),
            ("Hello, I need help", "Greeting Query"),
            ("What are the warranty terms?", "Warranty Query")
        ]
        
        for query, test_name in test_queries:
            test_results.append(await self.test_query_endpoint(query, test_name))
            await asyncio.sleep(1)  # Small delay between requests
        
        # Test 4: ASR Queries
        asr_tests = [
            ("um i need help with uh refund policy", "ASR with filler words"),
            ("can you tell me about shipping and delivery", "ASR shipping query"),
            ("hello i want to return something", "ASR return query")
        ]
        
        for asr_text, test_name in asr_tests:
            test_results.append(await self.test_asr_query(asr_text, test_name))
            await asyncio.sleep(1)
        
        # Test 5: Session History
        test_results.append(await self.test_session_history())
        
        # Test 6: System Stats
        test_results.append(await self.test_system_stats())
        
        # Summary
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        
        logger.info("=" * 60)
        logger.info(f"API TEST RESULTS: {passed_tests}/{total_tests} tests passed")
        logger.info("=" * 60)
        
        if passed_tests == total_tests:
            logger.info("✓ ALL API TESTS PASSED!")
            return True
        else:
            logger.error(f"✗ {total_tests - passed_tests} tests failed")
            return False
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

async def main():
    """Main test function."""
    tester = APITester()
    
    try:
        success = await tester.run_comprehensive_test()
        
        if success:
            print("\nNLP CRM System API is working correctly!")
        else:
            print("\nSome API tests failed. Check the logs for details.")
            
    except Exception as e:
        logger.error(f"Test execution error: {str(e)}")
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main())
