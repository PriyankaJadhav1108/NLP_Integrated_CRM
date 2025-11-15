# NLP CRM System

A comprehensive NLP-powered Customer Relationship Management (CRM) system with advanced RAG (Retrieval Augmented Generation) pipeline, hybrid search, intelligent chat management, and AI-driven customer interaction capabilities.

## 🚀 Features

### Core RAG Pipeline
- **Document Ingestion & Chunking**: Smart document processing with optimized text segmentation
- **Vector Store**: ChromaDB integration with persistent embeddings using HuggingFace models
- **Hybrid Search**: Combines BM25 keyword search with semantic vector search for optimal retrieval
- **Context Reranking**: Cross-encoder models for high-quality, context-aware result ranking

### Advanced NLP Capabilities
- **Query Transformation**: ASR text processing, query enhancement, and intent detection
- **Structured LLM Responses**: JSON-schema responses with confidence scores and source attribution
- **Chat History Management**: Persistent session management with conversation context
- **Sentiment Analysis**: Real-time sentiment detection and analysis
- **Entity Extraction**: Automatic extraction of entities (time, product, action, contact)

### CRM Integration
- **Interaction Logging**: Comprehensive logging of all customer interactions
- **Customer Profiles**: Advanced customer profiling with psychology traits and buying styles
- **Prescriptive AI**: AI-driven recommendations for customer interactions
- **Personality-Based Negotiation**: Cross-cultural negotiation engine with personality matching
- **Multilingual Support**: Multi-language negotiation and communication support

### API & Frontend
- **RESTful API**: FastAPI-based production-ready API server
- **Web Frontend**: Modern HTML/CSS/JavaScript frontend interface
- **Audio Processing**: Support for audio input with ASR (Automatic Speech Recognition)
- **Firebase Integration**: Optional cloud storage with Firestore

## 📁 Project Structure

```
nlpcrmzip/
└── nlpcrm/
    ├── main.py                    # Main entry point
    ├── config.py                  # Configuration settings
    ├── api_server.py              # FastAPI server with complete pipeline
    ├── crm_api.py                 # CRM-specific API endpoints
    ├── crm_models.py              # CRM data models
    ├── enhanced_nlp_crm.py        # Enhanced NLP analysis
    ├── document_processor.py      # Document ingestion and chunking
    ├── vector_store.py            # ChromaDB and embedding management
    ├── hybrid_search.py           # Hybrid search and reranking engine
    ├── chat_history.py            # Chat session and history management
    ├── llm_integration.py         # LLM integration and structured responses
    ├── query_transformer.py       # ASR processing and query transformation
    ├── personality_negotiator.py   # Personality-based negotiation
    ├── prescriptive_ai.py         # Prescriptive AI recommendations
    ├── multilingual_negotiator.py # Multilingual negotiation engine
    ├── product_db.py              # Product database management
    ├── frontend/                  # Web frontend files
    │   ├── index.html
    │   ├── app.js
    │   └── styles.css
    ├── crm_data/                  # CRM data storage
    │   ├── customers/
    │   ├── agents/
    │   └── interactions/
    ├── chat_history/              # Chat session storage
    ├── chroma_db/                 # Vector database storage
    ├── requirements.txt           # Python dependencies
    └── README.md                  # Detailed documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd nlpcrmzip
```

### Step 2: Install Dependencies
```bash
cd nlpcrm
pip install -r requirements.txt
```

### Step 3: Download SpaCy Model
```bash
python -m spacy download en_core_web_sm
```

### Step 4: Set Up Environment Variables
Create a `.env` file in the `nlpcrm` directory:

```env
# OpenAI API Key (optional, for real LLM integration)
OPENAI_API_KEY=your_openai_api_key_here

# ChromaDB Settings
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Application Settings
APP_NAME=NLP CRM API
APP_VERSION=1.0.0
DEBUG=True
API_KEY=nlpcrm-local-12345

# Firebase (optional)
FIRESTORE_ENABLED=False
FIREBASE_CREDENTIALS=path/to/service-account.json

# LLM Settings
USE_REAL_LLM=False
USE_OPENAI_WHISPER=True
```

## 🚦 Quick Start

### 1. Set Up the Knowledge Base
```bash
cd nlpcrm
python main.py --mode setup
```

### 2. Start the API Server
```bash
# Option 1: Using the provided batch file (Windows)
start_server.bat

# Option 2: Direct Python command
python api_server.py

# Option 3: Using uvicorn directly
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`

### 3. Access the Web Interface
Open your browser and navigate to:
- **API Documentation**: `http://localhost:8000/docs`
- **Frontend Interface**: `http://localhost:8000/frontend`
- **CRM Dashboard**: `http://localhost:8000/crm`

## 📖 Usage Examples

### Command Line Interface

```bash
# Set up knowledge base
python main.py --mode setup

# Interactive search
python main.py --mode search

# Run tests
python main.py --mode test

# Reset and set up fresh
python main.py --mode setup --reset
```

### API Usage

#### Create a Chat Session
```bash
curl -X POST "http://localhost:8000/v1/session" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "user123"}'
```

#### Ask a Question
```bash
curl -X POST "http://localhost:8000/v1/answer" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "How do I get a refund?",
       "session_id": "your-session-id",
       "asr_text": "um i need help with refund"
     }'
```

#### Get Conversation History
```bash
curl -X GET "http://localhost:8000/v1/session/{session_id}/history"
```

### Python API Usage

```python
from vector_store import VectorStoreManager
from chat_history import ChatHistoryManager
from llm_integration import LLMResponseGenerator

# Initialize components
vs_manager = VectorStoreManager()
chat_manager = ChatHistoryManager()
llm_generator = LLMResponseGenerator(use_mock=True)

# Set up knowledge base
vs_manager.setup_knowledge_base("customer_service_kb.md")

# Create a session
session = chat_manager.create_session(user_id="user123")

# Process a query
query = "How do I get a refund?"
retrieved_docs = vs_manager.hybrid_search_with_reranking(
    query=query,
    hybrid_k=10,
    final_k=3
)

# Generate response
response = llm_generator.generate_structured_response(
    query=query,
    retrieved_docs=retrieved_docs,
    conversation_history=chat_manager.get_conversation_history(session.session_id)
)

print(response.answer)
```

## 🔌 API Endpoints

### Core Endpoints
- `GET /` - Root endpoint with system information
- `GET /health` - Health check for all components
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

### Session Management
- `POST /v1/session` - Create a new chat session
- `GET /v1/session/{session_id}/history` - Get conversation history
- `DELETE /v1/session/{session_id}` - Delete a session

### Query Processing
- `POST /v1/answer` - Main query processing endpoint
- `POST /v1/answer/audio` - Process audio input with ASR

### CRM Endpoints
- `POST /api/v1/interactions` - Log customer interaction
- `GET /api/v1/interactions` - Get interaction logs
- `GET /api/v1/customers` - Get customer list
- `GET /api/v1/customers/{customer_id}` - Get customer details
- `POST /api/v1/customers` - Create/update customer profile

### System Information
- `GET /v1/stats` - Get system statistics
- `GET /v1/search/smoke-test` - Test search functionality

## 🧪 Testing

### Run Test Suites
```bash
# Phase 1 & 2 tests (RAG pipeline)
python test_retrieval.py

# Phase 4 component tests
python test_phase4_components.py

# API integration tests
python test_api_offline.py
python test_api.py

# CRM tests
python test_phase5_crm.py
```

## ⚙️ Configuration

Key configuration options in `config.py`:

- `CHUNK_SIZE`: Maximum size of text chunks (default: 1000 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200 characters)
- `CHROMA_PERSIST_DIRECTORY`: Directory for ChromaDB storage
- `MAX_CHUNKS`: Maximum number of chunks to process
- `USE_REAL_LLM`: Enable real LLM provider (OpenAI, etc.)
- `USE_OPENAI_WHISPER`: Use OpenAI Whisper for ASR
- `FIRESTORE_ENABLED`: Enable Firebase Firestore integration

## 📦 Dependencies

### Core Dependencies
- **fastapi**: Modern web framework for building APIs
- **uvicorn**: ASGI server for running FastAPI applications
- **langchain**: Document processing and vector store integration
- **chromadb**: Vector database for embeddings
- **sentence-transformers**: Text embedding generation
- **transformers**: Hugging Face transformers for NLP models
- **torch**: PyTorch for deep learning models

### Additional Dependencies
- **rank-bm25**: BM25 algorithm for keyword search
- **scikit-learn**: Machine learning utilities
- **spacy**: Advanced NLP processing
- **openai**: OpenAI API integration
- **openai-whisper**: Speech-to-text processing
- **firebase-admin**: Firebase integration (optional)
- **pydantic**: Data validation and settings management

See `requirements.txt` for the complete list of dependencies.

## 🔒 Security

- API key authentication support
- Environment variable configuration for sensitive data
- Secure session management
- Input validation and sanitization

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Download SpaCy model: `python -m spacy download en_core_web_sm`

2. **ChromaDB Issues**
   - Delete the `chroma_db` directory and restart if you encounter persistent issues
   - Check file permissions on the database directory

3. **Memory Issues**
   - The embedding model runs on CPU by default
   - For large documents, consider using a GPU-enabled environment
   - Reduce `MAX_CHUNKS` in config.py if needed

4. **File Not Found**
   - The system will create a sample knowledge base if `customer_service_kb.md` doesn't exist
   - Ensure all required directories exist (chat_history, crm_data, etc.)

5. **Port Already in Use**
   - Change the port in `api_server.py` or use: `uvicorn api_server:app --port 8001`

## 📝 Knowledge Base Format

The system expects a markdown file (`customer_service_kb.md`) with customer service information. The file should be structured with clear sections and subsections:

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
```

## 🚧 Future Enhancements

- Real-time document updates and synchronization
- Advanced analytics and reporting dashboard
- WebSocket support for real-time chat
- Multi-language knowledge base support
- Enhanced audio processing capabilities
- Integration with external CRM systems
- Performance optimization and caching
- Advanced security features

## 📄 License

This project is part of the NLP CRM system development.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues, questions, or contributions, please open an issue in the repository.

---

**Note**: This is a comprehensive NLP-powered CRM system designed for production use. Make sure to configure all environment variables and test thoroughly before deploying to production.
