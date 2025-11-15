"""
Hybrid Search and Reranking Module for NLP CRM System
Implements BM25 + Vector Search with Context Reranking
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain.schema import Document
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridSearchEngine:
    """Implements hybrid search combining BM25 and vector search with reranking."""
    
    def __init__(self, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the hybrid search engine.
        
        Args:
            reranker_model: Hugging Face model name for reranking
        """
        self.bm25_index = None
        self.documents = []
        self.tokenized_docs = []
        self.reranker = None
        self.reranker_model_name = reranker_model
        
        # Initialize reranker
        self._initialize_reranker()
        
        logger.info(f"Hybrid search engine initialized with reranker: {reranker_model}")
    
    def _initialize_reranker(self):
        """Initialize the reranker model."""
        try:
            self.reranker = CrossEncoder(self.reranker_model_name)
            logger.info(f"Reranker model loaded: {self.reranker_model_name}")
        except Exception as e:
            logger.error(f"Error loading reranker model: {str(e)}")
            # Fallback to a simpler model
            try:
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2")
                logger.info("Fallback reranker model loaded")
            except Exception as e2:
                logger.error(f"Error loading fallback reranker: {str(e2)}")
                self.reranker = None
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenization - can be enhanced with more sophisticated methods
        text = text.lower()
        # Remove special characters and split on whitespace
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def build_bm25_index(self, documents: List[Document]):
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of Document objects
        """
        try:
            self.documents = documents
            self.tokenized_docs = []
            
            for doc in documents:
                tokens = self._tokenize_text(doc.page_content)
                self.tokenized_docs.append(tokens)
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(self.tokenized_docs)
            
            logger.info(f"BM25 index built with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error building BM25 index: {str(e)}")
            raise
    
    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Perform BM25 search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if self.bm25_index is None:
            logger.error("BM25 index not built. Call build_bm25_index first.")
            return []
        
        try:
            # Tokenize query
            query_tokens = self._tokenize_text(query)
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top k results
            top_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include documents with positive scores
                    results.append((self.documents[idx], scores[idx]))
            
            logger.info(f"BM25 search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {str(e)}")
            return []
    
    def vector_search(self, query: str, vector_store, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Perform vector search using the existing vector store.
        
        Args:
            query: Search query
            vector_store: VectorStoreManager instance
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        try:
            results = vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Vector search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []
    
    def hybrid_search(self, query: str, vector_store, k: int = 10, 
                     bm25_weight: float = 0.3, vector_weight: float = 0.7) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Search query
            vector_store: VectorStoreManager instance
            k: Number of results to return
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector scores
            
        Returns:
            List of (document, hybrid_score) tuples
        """
        try:
            # Get results from both search methods
            bm25_results = self.bm25_search(query, k=k*2)  # Get more for better merging
            vector_results = self.vector_search(query, vector_store, k=k*2)
            
            # Create document to score mapping
            doc_scores = defaultdict(lambda: {'bm25': 0.0, 'vector': 0.0, 'doc': None})
            
            # Process BM25 results
            for doc, score in bm25_results:
                doc_id = id(doc.page_content)  # Use content hash as ID
                doc_scores[doc_id]['bm25'] = score
                doc_scores[doc_id]['doc'] = doc
            
            # Process vector results
            for doc, score in vector_results:
                doc_id = id(doc.page_content)
                doc_scores[doc_id]['vector'] = score
                doc_scores[doc_id]['doc'] = doc
            
            # Normalize scores
            bm25_scores = [doc_scores[doc_id]['bm25'] for doc_id in doc_scores]
            vector_scores = [doc_scores[doc_id]['vector'] for doc_id in doc_scores]
            
            if bm25_scores:
                max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
                min_bm25 = min(bm25_scores)
                bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1
            else:
                max_bm25 = min_bm25 = bm25_range = 1
            
            if vector_scores:
                max_vector = max(vector_scores) if max(vector_scores) > 0 else 1
                min_vector = min(vector_scores)
                vector_range = max_vector - min_vector if max_vector != min_vector else 1
            else:
                max_vector = min_vector = vector_range = 1
            
            # Calculate hybrid scores
            hybrid_results = []
            for doc_id, scores in doc_scores.items():
                # Normalize scores to [0, 1]
                norm_bm25 = (scores['bm25'] - min_bm25) / bm25_range if bm25_range > 0 else 0
                norm_vector = (scores['vector'] - min_vector) / vector_range if vector_range > 0 else 0
                
                # Calculate weighted hybrid score
                hybrid_score = (bm25_weight * norm_bm25) + (vector_weight * norm_vector)
                
                hybrid_results.append((scores['doc'], hybrid_score))
            
            # Sort by hybrid score and return top k
            hybrid_results.sort(key=lambda x: x[1], reverse=True)
            top_results = hybrid_results[:k]
            
            logger.info(f"Hybrid search returned {len(top_results)} results for query: {query[:50]}...")
            return top_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []
    
    def rerank_results(self, query: str, results: List[Tuple[Document, float]], 
                      top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Rerank search results using the cross-encoder model.
        
        Args:
            query: Search query
            results: List of (document, score) tuples
            top_k: Number of top results to return after reranking
            
        Returns:
            List of reranked (document, rerank_score) tuples
        """
        if self.reranker is None:
            logger.warning("Reranker not available, returning original results")
            return results[:top_k]
        
        if not results:
            return []
        
        try:
            # Prepare query-document pairs for reranking
            query_doc_pairs = []
            for doc, _ in results:
                query_doc_pairs.append([query, doc.page_content])
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(query_doc_pairs)
            
            # Combine with original results and rerank
            reranked_results = []
            for i, (doc, original_score) in enumerate(results):
                rerank_score = float(rerank_scores[i])
                reranked_results.append((doc, rerank_score))
            
            # Sort by rerank score and return top k
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            top_reranked = reranked_results[:top_k]
            
            logger.info(f"Reranked {len(results)} results to top {len(top_reranked)} for query: {query[:50]}...")
            return top_reranked
            
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            return results[:top_k]
    
    def hybrid_search_with_reranking(self, query: str, vector_store, 
                                   hybrid_k: int = 10, final_k: int = 3,
                                   bm25_weight: float = 0.3, vector_weight: float = 0.7) -> List[Tuple[Document, float]]:
        """
        Complete hybrid search pipeline with reranking.
        
        Args:
            query: Search query
            vector_store: VectorStoreManager instance
            hybrid_k: Number of results from hybrid search
            final_k: Number of final results after reranking
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector scores
            
        Returns:
            List of final reranked (document, score) tuples
        """
        try:
            # Step 1: Hybrid search with query + variations to improve recall
            hybrid_results = []
            candidate_queries = [query]
            try:
                # Leverage simple expansion: lowercased and without punctuation
                q_clean = re.sub(r"[^\w\s]", " ", query.lower())
                if q_clean and q_clean not in candidate_queries:
                    candidate_queries.append(q_clean)
            except Exception:
                pass
            for q in candidate_queries:
                partial = self.hybrid_search(
                    query=q,
                    vector_store=vector_store,
                    k=hybrid_k,
                    bm25_weight=bm25_weight,
                    vector_weight=vector_weight
                )
                hybrid_results.extend(partial)
            # Deduplicate by content
            seen = set()
            dedup = []
            for doc, score in hybrid_results:
                key = (doc.page_content[:100], doc.metadata.get('chunk_id'))
                if key in seen:
                    continue
                seen.add(key)
                dedup.append((doc, score))
            hybrid_results = dedup
            
            # Step 2: Rerank results
            final_results = self.rerank_results(
                query=query,
                results=hybrid_results,
                top_k=final_k
            )
            
            logger.info(f"Hybrid search with reranking completed: {len(final_results)} final results")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search with reranking: {str(e)}")
            return []
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the search engine."""
        stats = {
            "bm25_index_built": self.bm25_index is not None,
            "documents_indexed": len(self.documents),
            "reranker_available": self.reranker is not None,
            "reranker_model": self.reranker_model_name
        }
        return stats
