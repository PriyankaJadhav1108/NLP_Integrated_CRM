# NLP CRM System - Phase 1, 2 & 4: Complete RAG Pipeline with API

This project implements a complete NLP-powered CRM system with document ingestion, hybrid search, reranking, chat history management, LLM integration, and a production-ready API.

## Features

### Phase 1: Data Foundation & Core RAG

#### 1.1 Document Ingestion & Chunking
- **Document Processing**: Loads and processes markdown files (specifically `customer_service_kb.md`)
- **Smart Chunking**: Uses RecursiveCharacterTextSplitter with optimized separators for effective text segmentation
- **Metadata Management**: Attaches relevant metadata to each chunk for better organization

#### 1.2 Vector Store Setup
- **ChromaDB Integration**: Persistent vector database for storing document embeddings
- **Embedding Generation**: Uses HuggingFace's `all-MiniLM-L6-v2` model for efficient text vectorization
- **Semantic Retrieval**: Implements similarity search with relevance scoring

### Phase 2: Retrieval Robustness

#### 2.1 Hybrid Search
- **BM25 Index**: Implements BM25 (Best Matching 25) algorithm for keyword-based search
- **Vector Search**: Maintains semantic search capabilities using embeddings
- **Hybrid Fusion**: Combines BM25 and vector search results with configurable weights
- **Score Normalization**: Normalizes and merges scores from both search methods

#### 2.2 Context Reranking
- **Cross-Encoder Reranking**: Uses Hugging Face's cross-encoder models for context-aware reranking
- **High-Quality Results**: Reranks hybrid search results to return only the top 3 most relevant chunks
- **Configurable Pipeline**: Adjustable parameters for hybrid search and reranking stages

### Phase 4: Complete API Integration

#### 4.1 History Management Logic
- **Chat Session Management**: Persistent chat sessions with unique session IDs
- **Conversation History**: Automatic storage and retrieval of conversation context
- **Session Persistence**: JSON-based storage with automatic cleanup of old sessions
- **Multi-User Support**: User-specific session management

#### 4.2 LLM Structured Response
- **JSON Schema Response**: Standardized response format with confidence scores and metadata
- **Context-Aware Responses**: Uses conversation history and retrieved documents
- **Response Types**: Answer, clarification, escalation, greeting, goodbye, and error responses
- **Source Attribution**: Tracks and attributes information sources
- **Escalation Logic**: Automatic escalation detection for complex queries

#### 4.3 ASR & Query Transformation
- **ASR Text Processing**: Cleans and normalizes speech-to-text output
- **Query Enhancement**: Expands queries with synonyms and related terms
- **Intent Detection**: Identifies user intents (refund, shipping, support, etc.)
- **Query Variations**: Generates multiple query variations for better retrieval
- **Entity Extraction**: Extracts time, product, action, and contact entities

## Project Structure

```
nlpcrm/
├── main.py                    # Main entry point
├── config.py                 # Configuration settings
├── document_processor.py     # Document ingestion and chunking
├── vector_store.py           # ChromaDB and embedding management
├── hybrid_search.py          # Hybrid search and reranking engine
├── chat_history.py           # Chat session and history management
├── llm_integration.py        # LLM integration and structured responses
├── query_transformer.py      # ASR processing and query transformation
├── api_server.py             # FastAPI server with complete pipeline
├── test_retrieval.py         # Test suite for Phase 1 & 2
├── test_phase4_components.py # Test suite for Phase 4 components
├── test_api_offline.py       # Offline API testing
├── test_api.py               # Online API testing
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional):
   ```bash
   # Create a .env file with your settings
   OPENAI_API_KEY=your_openai_api_key_here
   CHROMA_PERSIST_DIRECTORY=./chroma_db
   DEBUG=True
   ```

## Usage

### Quick Start

1. **Set up the knowledge base**:
   ```bash
   python main.py --mode setup
   ```

2. **Test the system**:
   ```bash
   python main.py --mode test
   ```

3. **Interactive search**:
   ```bash
   python main.py --mode search
   ```

4. **Start the API server**:
   ```bash
   python api_server.py
   ```

5. **Test the complete system**:
   ```bash
   python test_phase4_components.py
   python test_api_offline.py
   ```

### Command Line Options

- `--mode {setup,search,test}`: Choose the operation mode
  - `setup`: Initialize the knowledge base (default)
  - `search`: Start interactive search interface
  - `test`: Run the test suite
- `--kb-file PATH`: Specify path to knowledge base file (default: `customer_service_kb.md`)
- `--reset`: Reset the vector store before setup

### Examples

```bash
# Set up with custom knowledge base file
python main.py --mode setup --kb-file my_kb.md

# Reset and set up fresh
python main.py --mode setup --reset

# Run tests
python main.py --mode test

# Interactive search
python main.py --mode search
```

## Knowledge Base Format

The system expects a markdown file (`customer_service_kb.md`) with customer service information. If the file doesn't exist, a sample file will be created automatically.

### Sample Knowledge Base Structure

```markdown
# Customer Service Knowledge Base

## General Policies
### Refund Policy
Our refund policy allows customers to return products within 30 days...

### Shipping Information
We offer free shipping on orders over $50...

## Product Support
### Technical Issues
For technical support, customers can contact...

## Account Management
### Password Reset
Customers can reset their password by...
```

## API Usage (Programmatic)

```python
from vector_store import VectorStoreManager
from document_processor import DocumentProcessor

# Initialize the system
vs_manager = VectorStoreManager()

# Set up knowledge base
vs_manager.setup_knowledge_base("customer_service_kb.md")

# Phase 1: Basic semantic search
results = vs_manager.similarity_search("How do I get a refund?", k=5)
results_with_scores = vs_manager.similarity_search_with_score("shipping policy", k=3)

# Phase 2: Hybrid search
hybrid_results = vs_manager.hybrid_search("refund policy 30 days", k=5)

# Phase 2: Hybrid search with reranking (recommended)
reranked_results = vs_manager.hybrid_search_with_reranking(
    query="How do I get a refund?",
    hybrid_k=10,  # Get 10 results from hybrid search
    final_k=3     # Return top 3 after reranking
)

# Get system statistics
stats = vs_manager.get_hybrid_search_stats()

# Phase 4: Complete API Integration
from chat_history import ChatHistoryManager
from llm_integration import LLMResponseGenerator
from query_transformer import QueryTransformer

# Initialize all components
vs_manager = VectorStoreManager()
chat_manager = ChatHistoryManager()
llm_generator = LLMResponseGenerator(use_mock=True)
query_transformer = QueryTransformer()

# Create a session
session = chat_manager.create_session(user_id="user123")

# Process a query with ASR
asr_text = "um i need help with uh refund policy"
cleaned_query = query_transformer.clean_asr_output(asr_text)
enhanced_query = query_transformer.enhance_query_for_retrieval(cleaned_query)

# Retrieve documents
retrieved_docs = vs_manager.hybrid_search_with_reranking(
    query=enhanced_query["cleaned_query"],
    hybrid_k=10,
    final_k=3
)

# Generate structured response
formatted_docs = [{"page_content": doc.page_content, "metadata": doc.metadata, "relevance_score": score} 
                  for doc, score in retrieved_docs]

response = llm_generator.generate_structured_response(
    query=cleaned_query,
    retrieved_docs=formatted_docs,
    conversation_history=chat_manager.get_conversation_history(session.session_id)
)

# Update chat history
chat_manager.add_message(session.session_id, "user", cleaned_query)
chat_manager.add_message(session.session_id, "assistant", response.answer)
```

## Configuration

Key configuration options in `config.py`:

- `CHUNK_SIZE`: Maximum size of text chunks (default: 1000 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200 characters)
- `CHROMA_PERSIST_DIRECTORY`: Directory for ChromaDB storage
- `MAX_CHUNKS`: Maximum number of chunks to process

## Testing

The system includes comprehensive tests for all phases:

### Phase 1 & 2 Tests
```bash
python test_retrieval.py
```

### Phase 4 Component Tests
```bash
python test_phase4_components.py
```

### API Integration Tests
```bash
python test_api_offline.py
```

Tests cover:
- Document processing and chunking
- Vector store initialization and hybrid search
- Knowledge base setup and semantic retrieval
- Chat history management
- LLM integration and structured responses
- Query transformation and ASR processing
- Complete API pipeline integration

## Dependencies

### Core Dependencies
- **langchain**: Document processing and vector store integration
- **chromadb**: Vector database for embeddings
- **sentence-transformers**: Text embedding generation
- **fastapi**: Web framework (for future API development)
- **pandas**: Data manipulation
- **markdown**: Markdown file processing

### Phase 2 Dependencies
- **rank-bm25**: BM25 algorithm implementation for keyword search
- **transformers**: Hugging Face transformers for reranking models
- **torch**: PyTorch for deep learning models
- **scikit-learn**: Machine learning utilities

### Phase 4 Dependencies
- **fastapi**: Modern web framework for building APIs
- **uvicorn**: ASGI server for running FastAPI applications
- **pydantic**: Data validation and settings management
- **python-multipart**: Support for multipart form data
- **httpx**: HTTP client for testing

## API Endpoints

The system provides a complete REST API:

### Core Endpoints
- `GET /` - Root endpoint with system information
- `GET /health` - Health check for all components
- `POST /v1/session` - Create a new chat session
- `POST /v1/answer` - Main query processing endpoint
- `GET /v1/session/{session_id}/history` - Get conversation history
- `GET /v1/stats` - Get system statistics

### Example API Usage
```bash
# Start the server
python api_server.py

# Create a session
curl -X POST "http://localhost:8000/v1/session" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "user123"}'

# Ask a question
curl -X POST "http://localhost:8000/v1/answer" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "How do I get a refund?",
       "session_id": "your-session-id",
       "asr_text": "um i need help with refund"
     }'
```

## Next Steps (Future Phases)

- Real LLM provider integration (OpenAI, Anthropic, etc.)
- Advanced query processing and intent classification
- Integration with external CRM systems
- Real-time document updates and synchronization
- Multi-language support and internationalization
- Performance optimization and caching
- Advanced analytics and reporting
- WebSocket support for real-time chat

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

2. **ChromaDB Issues**: Delete the `chroma_db` directory and restart if you encounter persistent issues

3. **Memory Issues**: The embedding model runs on CPU by default. For large documents, consider using a GPU-enabled environment

4. **File Not Found**: The system will create a sample knowledge base if `customer_service_kb.md` doesn't exist

### Logs

The system provides detailed logging. Check the console output for:
- Document processing status
- Vector store operations
- Search results and scores
- Error messages and debugging information

## License

This project is part of the NLP CRM system development.
