"""
Document Processing Module for NLP CRM System
Handles document ingestion, chunking, and preparation for vector storage.
"""

import os
import re
from typing import List, Dict, Any
from pathlib import Path
import markdown
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document ingestion and chunking for the CRM knowledge base."""
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter with custom separators for better chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n## ",    # Markdown H2 headers
                "\n### ",   # Markdown H3 headers  
                "\n#### ",  # Markdown H4 headers
                "\n\n",     # Double newlines
                "\n",       # Single newlines
                ". ",       # Sentences
                " ",        # Words
                ""          # Characters
            ]
        )
    
    def load_markdown_file(self, file_path: str) -> str:
        """
        Load and process a markdown file.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Processed text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Convert markdown to HTML then extract text
            html = markdown.markdown(content)
            
            # Remove HTML tags and clean up text
            text = re.sub(r'<[^>]+>', '', html)
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize whitespace
            text = text.strip()
            
            logger.info(f"Successfully loaded markdown file: {file_path}")
            return text
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def create_chunks(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Split text into chunks using the configured text splitter.
        
        Args:
            text: Text content to chunk
            metadata: Additional metadata to attach to each chunk
            
        Returns:
            List of Document objects
        """
        if metadata is None:
            metadata = {}
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = {
                **metadata,
                "chunk_id": i,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks)
            }
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        logger.info(f"Created {len(documents)} chunks from text")
        return documents
    
    def process_customer_service_kb(self, file_path: str = "customer_service_kb.md") -> List[Document]:
        """
        Process the customer service knowledge base file with section-based chunking.
        
        Args:
            file_path: Path to the customer service KB file
            
        Returns:
            List of processed document chunks
        """
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"Customer service KB file not found: {file_path}")
            # Create a sample file for demonstration
            self._create_sample_kb_file(file_path)
        
        # Load the file
        text_content = self.load_markdown_file(file_path)
        
        # Split into sections manually for better targeting
        documents = []
        sections = text_content.split('###')  # Split on H3 headers
        
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
                
            # Extract section title and content
            lines = section.split('\n', 1)
            title = lines[0].strip() if lines else f"Section {i}"
            content = lines[1].strip() if len(lines) > 1 else section
            
            if content:
                metadata = {
                    "source": file_path,
                    "document_type": "customer_service_kb",
                    "section_title": title,
                    "section_id": i,
                    "processed_at": str(pd.Timestamp.now())
                }
                
                # Create document for this section
                doc = Document(
                    page_content=f"{title}\n{content}",
                    metadata=metadata
                )
                documents.append(doc)
        
        logger.info(f"Processed customer service KB: {len(documents)} sections created")
        return documents
    
    def _create_sample_kb_file(self, file_path: str):
        """Create a sample customer service knowledge base file for demonstration."""
        sample_content = """# Customer Service Knowledge Base

## General Policies

### Refund Policy
Our refund policy allows customers to return products within 30 days of purchase. Refunds will be processed within 5-7 business days after we receive the returned item.

### Shipping Information
We offer free shipping on orders over $50. Standard shipping takes 3-5 business days, while express shipping takes 1-2 business days.

## Product Support

### Technical Issues
For technical support, customers can contact our support team via email at support@company.com or call our toll-free number at 1-800-SUPPORT.

### Warranty Information
All products come with a 1-year manufacturer warranty. Extended warranty options are available at purchase.

## Account Management

### Password Reset
Customers can reset their password by clicking the "Forgot Password" link on the login page. A reset link will be sent to their registered email address.

### Account Deactivation
To deactivate an account, customers should contact customer service. Account data will be retained for 90 days after deactivation.

## Billing and Payments

### Payment Methods
We accept all major credit cards, PayPal, and bank transfers. Payment is processed securely through our encrypted payment gateway.

### Billing Disputes
For billing disputes, customers should contact our billing department within 60 days of the charge. We will investigate and resolve disputes within 10 business days.

## Contact Information

### Customer Service Hours
Our customer service team is available Monday through Friday, 9 AM to 6 PM EST.

### Emergency Support
For urgent issues outside business hours, customers can use our emergency support line at 1-800-EMERGENCY.
"""
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(sample_content)
        
        logger.info(f"Created sample customer service KB file: {file_path}")

