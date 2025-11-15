import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # ChromaDB settings
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # Application settings
    APP_NAME = os.getenv("APP_NAME", "NLP CRM API")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    API_KEY = os.getenv("API_KEY", "")
    
    # Document processing settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_CHUNKS = 1000

    # Firestore
    FIRESTORE_ENABLED = os.getenv("FIRESTORE_ENABLED", "False").lower() == "true"
    FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS", "")  # path to service account json

    # LLM
    USE_REAL_LLM = os.getenv("USE_REAL_LLM", "False").lower() == "true"
    USE_OPENAI_WHISPER = os.getenv("USE_OPENAI_WHISPER", "True").lower() == "true"
