"""
FastAPI Server for NLP CRM System
Main API endpoint that integrates all components.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field
import asyncio
import uvicorn

# Import our modules
from vector_store import VectorStoreManager
from chat_history import ChatHistoryManager, ChatSession
from llm_integration import LLMResponseGenerator, StructuredResponse, InformationSource, SourceType, ConfidenceLevel
from query_transformer import QueryTransformer
from crm_api import router as crm_router
from crm_api import (
    get_crm_storage,
    _determine_intent_from_response,
    _determine_sentiment_from_query,
    _determine_interaction_type,
)
from personality_negotiator import (
    CrossCulturalNegotiator,
    CustomerProfile,
    NegotiationContext,
    PersonalityType,
    CulturalRegion
)
from prescriptive_ai import (
    PrescriptiveAI,
    CustomerProfile as PrescriptiveCustomerProfile,
    InteractionHistory,
    PrescriptiveRecommendation,
    PsychologyTrait,
    BuyingStyle,
    ActionType
)
from multilingual_negotiator import (
    MultilingualNegotiationEngine,
    MultilingualProfile,
    TranscriptSegment,
    Language,
    PersonalityTrait as MultilingualPersonalityTrait,
    CulturalContext,
    NegotiationRecommendation as MultilingualNegotiationRecommendation
)
from enhanced_nlp_crm import get_enhanced_nlp_crm
from product_db import ProductDB
from crm_models import InteractionLog, InteractionStatus
from config import Config
from io import BytesIO
import tempfile
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NLP CRM System API",
    description="Intelligent Customer Service API with RAG and LLM Integration",
    version="1.0.0"
)

# Ensure bundled FFmpeg is available on PATH for local Whisper ASR
def _ensure_ffmpeg_on_path() -> None:
    try:
        ffmpeg_dir = None
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ffmpeg_root = os.path.join(base_dir, "ffmpeg")
        if os.path.isdir(ffmpeg_root):
            # Find any nested ffmpeg build's bin directory
            for entry in os.listdir(ffmpeg_root):
                candidate = os.path.join(ffmpeg_root, entry, "bin")
                if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "ffmpeg.exe")):
                    ffmpeg_dir = candidate
                    break
        if ffmpeg_dir and ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}" + os.environ.get("PATH", "")
            logger.info(f"FFmpeg added to PATH from: {ffmpeg_dir}")
    except Exception as e:
        logger.warning(f"Failed to ensure FFmpeg on PATH: {str(e)}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include CRM router
app.include_router(crm_router, dependencies=[])

# Serve frontend static assets
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Global instances
vector_store_manager = None
chat_history_manager = None
llm_response_generator = None
query_transformer = None
product_db = None
cross_cultural_negotiator = None
prescriptive_ai = None
multilingual_negotiator = None

# High-risk legal/compliance terms (fast exact/substring matching)
LEGAL_KEYWORDS = {
    "lawsuit", "attorney", "lawyer", "litigation", "class action", "subpoena",
    "cease and desist", "demand letter", "breach of", "negligence",
}
REGULATORY_BODIES = {
    "fcc", "ftc", "gdpr", "hipaa", "ofcom", "ico", "cpra", "ccpa",
    "sec", "fca", "ocl", "oig", "cma", "doj"
}

TOXICITY_TERMS = {
    # abusive/insulting
    "idiot", "stupid", "moron", "useless", "trash", "garbage",
    # threats
    "kill", "beat", "hurt", "threaten", "sue you", "destroy",
    # hateful (simple list; not exhaustive)
    "racist", "sexist"
}

def detect_legal_compliance_issue(text: str) -> Optional[Dict[str, Any]]:
    """Return match info if text indicates legal/compliance trigger."""
    if not text:
        return None
    t = text.lower()
    matched: Dict[str, Any] = {"keywords": [], "regulators": []}
    for kw in LEGAL_KEYWORDS:
        if kw in t:
            matched["keywords"].append(kw)
    for rb in REGULATORY_BODIES:
        if rb in t:
            matched["regulators"].append(rb.upper())
    if matched["keywords"] or matched["regulators"]:
        reason_parts = []
        if matched["keywords"]:
            reason_parts.append(f"keywords: {', '.join(sorted(set(matched['keywords'])))}")
        if matched["regulators"]:
            reason_parts.append(f"regulators: {', '.join(sorted(set(matched['regulators'])))}")
        return {
            "reason": "High-risk legal/compliance trigger detected (" + "; ".join(reason_parts) + ")",
            "queue": "Legal/Compliance",
            **matched,
        }
    return None

def detect_toxicity(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.lower()
    found = [term for term in TOXICITY_TERMS if term in t]
    return {"terms": sorted(set(found))} if found else None

# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="User's question or request")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    user_id: Optional[str] = Field(None, description="User identifier")
    asr_text: Optional[str] = Field(None, description="ASR transcription (if available)")
    include_history: bool = Field(True, description="Include conversation history")
    max_history_messages: int = Field(10, description="Maximum history messages to include")

class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    success: bool
    response: Optional[StructuredResponse] = None
    session_id: Optional[str] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    query_metadata: Optional[Dict[str, Any]] = None

class SessionRequest(BaseModel):
    """Request model for creating a new session."""
    user_id: Optional[str] = Field(None, description="User identifier")

class SessionResponse(BaseModel):
    """Response model for session creation."""
    success: bool
    session_id: Optional[str] = None
    error_message: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    components: Dict[str, str]

class RiskRequest(BaseModel):
    text: str = Field(..., description="Raw customer transcript to analyze")

class RiskResponse(BaseModel):
    is_toxic_flagged: bool
    is_compliance_flagged: bool
    toxicity_terms: Optional[List[str]] = None
    compliance_terms: Optional[List[str]] = None

class NegotiatorRequest(BaseModel):
    customer_id: str
    text: str = Field(..., description="Customer communication text to analyze")
    context: str = Field(..., description="Negotiation context (sales, support, partnership)")
    customer_profile: Optional[Dict[str, Any]] = None
    negotiation_context: Optional[Dict[str, Any]] = None

class NegotiatorResponse(BaseModel):
    customer_id: str
    context: str
    detected_personality: str
    culture: str
    recommended_message: str
    justification: str
    negotiation_strategy: str
    tone_guidelines: Dict[str, Any]
    cultural_considerations: List[str]
    alternative_approaches: List[str]
    confidence_score: float

class PrescriptiveRequest(BaseModel):
    customer_id: str
    customer_profile: Optional[Dict[str, Any]] = None
    interaction_history: Optional[List[Dict[str, Any]]] = None

class PrescriptiveResponse(BaseModel):
    customer_id: str
    conversion_probability: float
    churn_risk: float
    recommended_action: str
    action_type: str
    suggested_timeframe: str
    justification: str
    confidence_score: float
    expected_outcome: str
    priority: str
    required_resources: List[str]
    alternative_actions: List[str]
    success_metrics: Dict[str, Any]
    psychology_traits: List[str]
    buying_style: Optional[str] = None

class MultilingualNegotiationRequest(BaseModel):
    customer_id: str
    context: str
    audio_file_path: Optional[str] = None
    transcript: Optional[List[Dict[str, Any]]] = None
    customer_profile: Optional[Dict[str, Any]] = None

class MultilingualNegotiationResponse(BaseModel):
    customer_id: str
    context: str
    transcript: List[Dict[str, Any]]
    detected_language: str
    detected_personality: str
    culture: str
    recommended_message: str
    recommended_message_hindi: Optional[str] = None
    recommended_message_marathi: Optional[str] = None
    justification: str
    negotiation_strategy: str
    cultural_considerations: List[str]
    tone_guidelines: Dict[str, Any]
    confidence_score: float

class EnhancedNLPAnalysisResponse(BaseModel):
    success: bool
    analysis: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class EnhancedNLPRequest(BaseModel):
    text: str = Field(..., description="Raw text to analyze (e.g., ASR transcript)")

# Dependency to get initialized components
def get_components():
    """Get initialized components."""
    global vector_store_manager, chat_history_manager, llm_response_generator, query_transformer
    
    if not all([vector_store_manager, chat_history_manager, llm_response_generator, query_transformer]):
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "vector_store": vector_store_manager,
        "chat_history": chat_history_manager,
        "llm_generator": llm_response_generator,
        "query_transformer": query_transformer
    }

# Simple API key security dependency
def require_api_key(x_api_key: str = Header(None)):
    if Config.API_KEY:
        if not x_api_key or x_api_key != Config.API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# API Endpoints
@app.get("/")
async def root():
    """Redirect root to CRM dashboard for convenience."""
    return RedirectResponse(url="/crm")

@app.get("/info", response_model=Dict[str, str])
async def info():
    """API info endpoint (previous root JSON)."""
    return {
        "message": "NLP CRM System API",
        "version": "1.0.0",
        "status": "running",
        "crm_dashboard": "/crm"
    }

@app.get("/crm")
async def crm_dashboard():
    """Serve the CRM dashboard frontend."""
    return FileResponse("frontend/index.html")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components = {}
    
    try:
        # Check vector store
        if vector_store_manager:
            info = vector_store_manager.get_collection_info()
            components["vector_store"] = "healthy" if info.get("document_count", 0) > 0 else "no_documents"
        else:
            components["vector_store"] = "not_initialized"
    except Exception as e:
        components["vector_store"] = f"error: {str(e)}"
    
    try:
        # Check chat history
        if chat_history_manager:
            stats = chat_history_manager.get_session_stats()
            components["chat_history"] = "healthy"
        else:
            components["chat_history"] = "not_initialized"
    except Exception as e:
        components["chat_history"] = f"error: {str(e)}"
    
    try:
        # Check LLM generator
        if llm_response_generator:
            components["llm_generator"] = "healthy"
        else:
            components["llm_generator"] = "not_initialized"
    except Exception as e:
        components["llm_generator"] = f"error: {str(e)}"
    
    try:
        # Check query transformer
        if query_transformer:
            components["query_transformer"] = "healthy"
        else:
            components["query_transformer"] = "not_initialized"
    except Exception as e:
        components["query_transformer"] = f"error: {str(e)}"
    
    overall_status = "healthy" if all("error" not in status and "not_initialized" not in status 
                                    for status in components.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        components=components
    )

@app.post("/v1/session", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """Create a new chat session."""
    try:
        components = get_components()
        chat_history = components["chat_history"]
        
        session = chat_history.create_session(user_id=request.user_id)
        
        return SessionResponse(
            success=True,
            session_id=session.session_id
        )
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        return SessionResponse(
            success=False,
            error_message=str(e)
        )

@app.post("/v1/answer", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
async def answer_query(request: QueryRequest):
    """
    Main endpoint for answering customer queries.
    Integrates ASR, query transformation, retrieval, and LLM response generation.
    """
    start_time = time.time()
    
    try:
        components = get_components()
        vector_store = components["vector_store"]
        chat_history = components["chat_history"]
        llm_generator = components["llm_generator"]
        query_transformer = components["query_transformer"]
        # Local product DB
        global product_db
        if product_db is None:
            product_db = ProductDB()
        
        # Step 1: Handle session
        session_id = request.session_id
        if not session_id:
            # Create new session if none provided
            session = chat_history.create_session(user_id=request.user_id)
            session_id = session.session_id
        else:
            # Verify session exists
            session = chat_history.get_session(session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        
        # Step 2: Process query (ASR + transformation)
        raw_query = request.asr_text if request.asr_text else request.query
        
        # Transform and enhance query
        query_metadata = query_transformer.get_query_metadata(raw_query)
        enhanced_query = query_transformer.enhance_query_for_retrieval(raw_query)
        
        # Use the best query variation for retrieval
        retrieval_queries = query_transformer.transform_for_retrieval(raw_query, max_variations=3)
        primary_query = retrieval_queries[0] if retrieval_queries else raw_query
        
        # Optional: Immediate legal/compliance escalation (preemptive)
        legal_hit = detect_legal_compliance_issue(raw_query)

        # Step 3: Retrieve relevant documents
        retrieved_docs = vector_store.hybrid_search_with_reranking(
            query=primary_query,
            hybrid_k=10,
            final_k=3
        )
        # Fallback: if nothing returned or an intermediate error, try plain vector similarity
        if not retrieved_docs:
            try:
                sim = vector_store.similarity_search_with_score(primary_query, k=3)
                retrieved_docs = [(doc, float(score)) for doc, score in sim]
                logger.info("Fallback to vector similarity yielded %d docs", len(retrieved_docs))
            except Exception as e:
                logger.warning(f"Vector fallback failed: {str(e)}")
        
        # Convert to format expected by LLM
        formatted_docs = []
        for doc, score in retrieved_docs:
            formatted_docs.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score
            })

        # Step 3b: Enrich with structured ProductDB facts if SKU detected
        sku_from_entities = None
        try:
            entities = query_metadata.get("extracted_entities", {}) if isinstance(query_metadata, dict) else {}
            product_entities = entities.get("product_entities", [])
            for ent in product_entities:
                ent_upper = str(ent).upper()
                if ent_upper.startswith("SKU"):
                    sku_from_entities = ent_upper
                    break
        except Exception:
            sku_from_entities = None

        structured_facts = None
        if sku_from_entities:
            structured_facts = product_db.build_definitive_facts(sku_from_entities)
            if structured_facts:
                formatted_docs.insert(0, {
                    "page_content": "Definitive structured product facts",
                    "metadata": {**structured_facts, "source": "structured_product_db"},
                    "relevance_score": 1.0
                })
        
        # Step 4: Get conversation history
        conversation_history = None
        if request.include_history:
            conversation_history = chat_history.get_conversation_history(
                session_id=session_id,
                max_messages=request.max_history_messages
            )
        
        # Step 5: Generate structured LLM response (or legal escalation)
        if legal_hit:
            from llm_integration import StructuredResponse, ResponseType, ConfidenceLevel, InformationSource, SourceType
            structured_response = StructuredResponse(
                response_type=ResponseType.ESCALATION,
                answer=(
                    "Your message indicates a potential legal or regulatory matter. "
                    "I am escalating this interaction to our Legal/Compliance team immediately. "
                    "They will reach out to you shortly."
                ),
                confidence=ConfidenceLevel.HIGH,
                sources=[
                    InformationSource(
                        source_type=SourceType.FALLBACK,
                        content="Automatic legal/compliance trigger",
                        relevance_score=1.0,
                        metadata={"matched": legal_hit}
                    )
                ],
                escalation_required=True,
                escalation_reason=legal_hit["reason"],
                suggested_actions=["Assign to Legal/Compliance queue", "Review customer contract and transcripts"],
                follow_up_questions=["Please provide any case numbers or prior notices received."]
            )
        else:
            structured_response = llm_generator.generate_structured_response(
                query=raw_query,
                retrieved_docs=formatted_docs,
                conversation_history=conversation_history,
                query_metadata=query_metadata
            )

        # Remove safety net that was causing KB dumps

        # Ensure structured facts are visible in the response sources
        if structured_facts:
            try:
                structured_source = InformationSource(
                    source_type=SourceType.EXTERNAL_API,
                    content="Structured product database facts",
                    relevance_score=1.0,
                    metadata={**structured_facts, "source": "structured_product_db"}
                )
                structured_response.sources.insert(0, structured_source)
                # Optionally bump confidence when definitive facts exist
                if structured_response.confidence == ConfidenceLevel.MEDIUM:
                    structured_response.confidence = ConfidenceLevel.HIGH
            except Exception:
                pass
        
        # Step 6: Update chat history
        chat_history.add_message(session_id, "user", raw_query)
        chat_history.add_message(session_id, "assistant", structured_response.answer)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        structured_response.processing_time_ms = processing_time
        
        # Add query metadata to response
        query_metadata["processing_time_ms"] = processing_time
        query_metadata["retrieved_docs_count"] = len(formatted_docs)
        query_metadata["session_id"] = session_id
        
        logger.info(f"Successfully processed query in {processing_time}ms for session {session_id}")
        
        # Step 7: Auto-log interaction to CRM
        try:
            storage = get_crm_storage()
            # Derive intent and sentiment using CRM helpers
            intent = _determine_intent_from_response(str(structured_response.response_type), raw_query)
            sentiment = _determine_sentiment_from_query(raw_query)
            interaction_type = _determine_interaction_type(intent, structured_response.escalation_required)
            status = InteractionStatus.ESCALATED if structured_response.escalation_required else InteractionStatus.PENDING

            interaction = InteractionLog(
                status=status,
                interaction_type=interaction_type,
                intent=intent,
                sentiment=sentiment,
                customer_query=raw_query,
                assistant_response=structured_response.answer,
                response_type=structured_response.response_type,
                confidence=structured_response.confidence,
                metadata={
                    "session_id": session_id,
                    "user_id": request.user_id,
                    "timestamp": datetime.now().isoformat(),
                    "processing_time_ms": processing_time,
                    "query_metadata": query_metadata,
                    "retrieved_docs_count": len(formatted_docs),
                    "confidence_score": None,
                    "escalation_reason": structured_response.escalation_reason,
                    "source_channel": "web_api",
                },
                sources=[s.dict() if hasattr(s, 'dict') else s for s in structured_response.sources],
                conversation_context=conversation_history,
                customer_id=request.user_id,
                agent_id=None,
                department=("Legal/Compliance" if legal_hit else None),
                priority=1,
                tags=((query_metadata.get("detected_intents", []) if isinstance(query_metadata, dict) else []) + (["legal_risk"] if legal_hit else [])),
                suggested_actions=structured_response.suggested_actions,
                follow_up_questions=structured_response.follow_up_questions,
                escalation_required=structured_response.escalation_required,
                user_satisfaction_prediction=structured_response.user_satisfaction_prediction,
                response_quality_score=structured_response.response_quality_score,
            )
            storage.log_interaction(interaction)
        except Exception as e:
            logger.warning(f"Auto CRM logging failed: {str(e)}")

        return QueryResponse(
            success=True,
            response=structured_response,
            session_id=session_id,
            processing_time_ms=processing_time,
            query_metadata=query_metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"Error processing query: {str(e)}")
        
        return QueryResponse(
            success=False,
            error_message=str(e),
            processing_time_ms=processing_time,
            query_metadata={"error": str(e)}
        )

@app.get("/v1/session/{session_id}/history", dependencies=[Depends(require_api_key)])
async def get_session_history(session_id: str, limit: int = 20):
    """Get conversation history for a session."""
    try:
        components = get_components()
        chat_history = components["chat_history"]
        
        session = chat_history.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = chat_history.get_recent_messages(session_id, limit=limit)
        
        return {
            "success": True,
            "session_id": session_id,
            "messages": messages,
            "total_messages": len(session.messages)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/stats", dependencies=[Depends(require_api_key)])
async def get_system_stats():
    """Get system statistics."""
    try:
        components = get_components()
        
        stats = {
            "timestamp": datetime.now().isoformat(),
            "vector_store": components["vector_store"].get_collection_info(),
            "chat_history": components["chat_history"].get_session_stats(),
            "hybrid_search": components["vector_store"].get_hybrid_search_stats()
        }
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/risk/classify", response_model=RiskResponse, dependencies=[Depends(require_api_key)])
async def classify_risk(request: RiskRequest):
    """Risk Classification Engine - strict JSON output."""
    text = request.text or ""
    tox = detect_toxicity(text)
    comp = detect_legal_compliance_issue(text)
    return RiskResponse(
        is_toxic_flagged=bool(tox),
        is_compliance_flagged=bool(comp),
        toxicity_terms=(tox.get("terms") if tox else []),
        compliance_terms=((comp.get("keywords", []) + comp.get("regulators", [])) if comp else [])
    )

@app.post("/v1/negotiator/analyze", response_model=NegotiatorResponse, dependencies=[Depends(require_api_key)])
async def analyze_negotiation(request: NegotiatorRequest):
    """Cross-Cultural, Personality-Aware AI Negotiator - Advanced CRM feature."""
    try:
        global cross_cultural_negotiator
        if cross_cultural_negotiator is None:
            cross_cultural_negotiator = CrossCulturalNegotiator()
        
        # Create customer profile from request data
        customer_profile = CustomerProfile(
            customer_id=request.customer_id,
            age_range=request.customer_profile.get("age_range") if request.customer_profile else None,
            region=request.customer_profile.get("region") if request.customer_profile else None,
            culture=request.customer_profile.get("culture") if request.customer_profile else None,
            personality_traits=[],
            communication_preferences=request.customer_profile.get("communication_preferences", {}) if request.customer_profile else {},
            past_interactions=request.customer_profile.get("past_interactions", []) if request.customer_profile else [],
            negotiation_history=request.customer_profile.get("negotiation_history", []) if request.customer_profile else []
        )
        
        # Create negotiation context
        negotiation_context = NegotiationContext(
            intent=request.context,
            product_service=request.negotiation_context.get("product_service", "General") if request.negotiation_context else "General",
            current_stage=request.negotiation_context.get("current_stage", "initial_contact") if request.negotiation_context else "initial_contact",
            customer_objections=request.negotiation_context.get("customer_objections", []) if request.negotiation_context else [],
            budget_range=request.negotiation_context.get("budget_range") if request.negotiation_context else None,
            timeline=request.negotiation_context.get("timeline") if request.negotiation_context else None,
            decision_makers=request.negotiation_context.get("decision_makers", []) if request.negotiation_context else []
        )
        
        # Generate recommendation
        recommendation = cross_cultural_negotiator.generate_recommendation(
            customer_profile, 
            negotiation_context, 
            request.text
        )
        
        return NegotiatorResponse(
            customer_id=recommendation.customer_id,
            context=recommendation.context,
            detected_personality=recommendation.detected_personality,
            culture=recommendation.culture,
            recommended_message=recommendation.recommended_message,
            justification=recommendation.justification,
            negotiation_strategy=recommendation.negotiation_strategy,
            tone_guidelines=recommendation.tone_guidelines,
            cultural_considerations=recommendation.cultural_considerations,
            alternative_approaches=recommendation.alternative_approaches,
            confidence_score=recommendation.confidence_score
        )
        
    except Exception as e:
        logger.error(f"Negotiator analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Negotiation analysis failed: {str(e)}")

@app.post("/v1/prescriptive/analyze", response_model=PrescriptiveResponse, dependencies=[Depends(require_api_key)])
async def analyze_prescriptive(request: PrescriptiveRequest):
    """Prescriptive AI - Predictive & Prescriptive Intelligence for advanced CRM."""
    try:
        global prescriptive_ai
        if prescriptive_ai is None:
            prescriptive_ai = PrescriptiveAI()
        
        # Create customer profile from request data
        customer_profile = PrescriptiveCustomerProfile(
            customer_id=request.customer_id,
            demographics=request.customer_profile.get("demographics", {}) if request.customer_profile else {},
            preferences=request.customer_profile.get("preferences", {}) if request.customer_profile else {},
            sentiment_score=request.customer_profile.get("sentiment_score", 0.0) if request.customer_profile else 0.0,
            lead_score=request.customer_profile.get("lead_score", 0.0) if request.customer_profile else 0.0,
            last_interaction=datetime.fromisoformat(request.customer_profile.get("last_interaction")) if request.customer_profile and request.customer_profile.get("last_interaction") else None,
            total_interactions=request.customer_profile.get("total_interactions", 0) if request.customer_profile else 0,
            successful_deals=request.customer_profile.get("successful_deals", 0) if request.customer_profile else 0,
            avg_deal_value=request.customer_profile.get("avg_deal_value", 0.0) if request.customer_profile else 0.0,
            response_rate=request.customer_profile.get("response_rate", 0.0) if request.customer_profile else 0.0,
            preferred_communication=request.customer_profile.get("preferred_communication", "email") if request.customer_profile else "email",
            timezone=request.customer_profile.get("timezone", "UTC") if request.customer_profile else "UTC"
        )
        
        # Create interaction history from request data
        interaction_history = []
        if request.interaction_history:
            for interaction_data in request.interaction_history:
                interaction = InteractionHistory(
                    interaction_id=interaction_data.get("interaction_id", ""),
                    customer_id=request.customer_id,
                    interaction_type=interaction_data.get("interaction_type", "email"),
                    timestamp=datetime.fromisoformat(interaction_data.get("timestamp", datetime.now().isoformat())),
                    content=interaction_data.get("content", ""),
                    sentiment=interaction_data.get("sentiment", 0.0),
                    outcome=interaction_data.get("outcome", "neutral"),
                    response_time_hours=interaction_data.get("response_time_hours", 24.0),
                    engagement_level=interaction_data.get("engagement_level", 0.5),
                    offer_presented=interaction_data.get("offer_presented"),
                    discount_offered=interaction_data.get("discount_offered"),
                    decision_made=interaction_data.get("decision_made"),
                    deal_value=interaction_data.get("deal_value")
                )
                interaction_history.append(interaction)
        
        # Generate prescriptive recommendation
        recommendation = prescriptive_ai.analyze_customer(customer_profile, interaction_history)
        
        return PrescriptiveResponse(
            customer_id=recommendation.customer_id,
            conversion_probability=recommendation.conversion_probability,
            churn_risk=recommendation.churn_risk,
            recommended_action=recommendation.recommended_action,
            action_type=recommendation.action_type.value,
            suggested_timeframe=recommendation.suggested_timeframe,
            justification=recommendation.justification,
            confidence_score=recommendation.confidence_score,
            expected_outcome=recommendation.expected_outcome,
            priority=recommendation.priority,
            required_resources=recommendation.required_resources,
            alternative_actions=recommendation.alternative_actions,
            success_metrics=recommendation.success_metrics,
            psychology_traits=[trait.value for trait in customer_profile.psychology_traits],
            buying_style=customer_profile.buying_style.value if customer_profile.buying_style else None
        )
        
    except Exception as e:
        logger.error(f"Prescriptive AI analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prescriptive AI analysis failed: {str(e)}")

@app.post("/v1/multilingual/negotiate", response_model=MultilingualNegotiationResponse, dependencies=[Depends(require_api_key)])
async def analyze_multilingual_negotiation(request: MultilingualNegotiationRequest):
    """Multilingual AI Negotiator - Cross-Cultural, Personality-Aware with multilingual transcription."""
    try:
        global multilingual_negotiator
        if multilingual_negotiator is None:
            multilingual_negotiator = MultilingualNegotiationEngine()
        
        # Create multilingual profile from request data
        customer_profile = MultilingualProfile(
            customer_id=request.customer_id,
            primary_language=Language(request.customer_profile.get("primary_language", "en")) if request.customer_profile else Language.ENGLISH,
            cultural_context=CulturalContext(request.customer_profile.get("cultural_context", "indian_urban")) if request.customer_profile else CulturalContext.INDIAN_URBAN,
            personality_traits=[],
            communication_style=request.customer_profile.get("communication_style", "mixed") if request.customer_profile else "mixed",
            negotiation_preferences=request.customer_profile.get("negotiation_preferences", {}) if request.customer_profile else {}
        )
        
        # Create transcript from request data or use audio file
        if request.transcript:
            transcript_segments = []
            for segment_data in request.transcript:
                segment = TranscriptSegment(
                    speaker=segment_data.get("speaker", "Unknown"),
                    text=segment_data.get("text", ""),
                    language=Language(segment_data.get("language", "en")),
                    confidence=segment_data.get("confidence", 0.8),
                    timestamp=segment_data.get("timestamp", 0.0),
                    sentiment=segment_data.get("sentiment", 0.0)
                )
                transcript_segments.append(segment)
        else:
            # Use audio file for transcription
            audio_file_path = request.audio_file_path or "sample_audio.wav"
            transcript_segments = multilingual_negotiator.stt.transcribe_audio(audio_file_path)
        
        # Generate multilingual negotiation recommendation
        recommendation = multilingual_negotiator.analyze_negotiation(
            request.audio_file_path or "sample_audio.wav", 
            customer_profile, 
            request.context
        )
        
        # Convert transcript to dict format for response
        transcript_dict = []
        for segment in recommendation.transcript:
            transcript_dict.append({
                "speaker": segment.speaker,
                "text": segment.text,
                "language": segment.language.value,
                "confidence": segment.confidence,
                "timestamp": segment.timestamp,
                "sentiment": segment.sentiment
            })
        
        return MultilingualNegotiationResponse(
            customer_id=recommendation.customer_id,
            context=recommendation.context,
            transcript=transcript_dict,
            detected_language=recommendation.detected_language,
            detected_personality=recommendation.detected_personality,
            culture=recommendation.culture,
            recommended_message=recommendation.recommended_message,
            recommended_message_hindi=recommendation.recommended_message_hindi,
            recommended_message_marathi=recommendation.recommended_message_marathi,
            justification=recommendation.justification,
            negotiation_strategy=recommendation.negotiation_strategy,
            cultural_considerations=recommendation.cultural_considerations,
            tone_guidelines=recommendation.tone_guidelines,
            confidence_score=recommendation.confidence_score
        )
        
    except Exception as e:
        logger.error(f"Multilingual negotiation analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multilingual negotiation analysis failed: {str(e)}")

@app.post("/v1/enhanced-nlp/analyze", response_model=EnhancedNLPAnalysisResponse, dependencies=[Depends(require_api_key)])
async def analyze_enhanced_nlp(request: EnhancedNLPRequest):
    """Enhanced NLP Analysis - Comprehensive text analysis with sentiment, entities, summarization, and priority scoring."""
    try:
        enhanced_nlp = get_enhanced_nlp_crm()
        
        # Perform comprehensive analysis
        analysis_results = enhanced_nlp.analyze_text(request.text)
        
        return EnhancedNLPAnalysisResponse(
            success=True,
            analysis=analysis_results
        )
        
    except Exception as e:
        logger.error(f"Enhanced NLP analysis failed: {str(e)}")
        return EnhancedNLPAnalysisResponse(
            success=False,
            error_message=f"Enhanced NLP analysis failed: {str(e)}"
        )

@app.post("/v1/multilingual/transcribe", dependencies=[Depends(require_api_key)])
async def transcribe_multilingual_audio(file: UploadFile = File(...)):
    """Transcribe multilingual audio file and return transcript with language detection."""
    try:
        global multilingual_negotiator
        if multilingual_negotiator is None:
            multilingual_negotiator = MultilingualNegotiationEngine()
        
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe audio
            transcript_segments = multilingual_negotiator.stt.transcribe_audio(temp_file_path)
            
            # Convert to dict format for response
            transcript_dict = []
            for segment in transcript_segments:
                transcript_dict.append({
                    "speaker": segment.speaker,
                    "text": segment.text,
                    "language": segment.language.value,
                    "confidence": segment.confidence,
                    "timestamp": segment.timestamp,
                    "sentiment": segment.sentiment
                })
            
            return {
                "success": True,
                "transcript": transcript_dict,
                "message": "Audio transcribed successfully"
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except Exception as e:
        logger.error(f"Audio transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {str(e)}")

@app.post("/v1/nlp/enhanced", response_model=EnhancedNLPAnalysisResponse, dependencies=[Depends(require_api_key)])
async def enhanced_nlp_analysis(request: EnhancedNLPRequest):
    """Lightweight enhanced NLP analysis backed by existing QueryTransformer.
    Extracts language, intents, sentiment, priority, and entities in one call.
    """
    try:
        # Prefer existing global instance; lazily initialize if needed
        global query_transformer
        if query_transformer is None:
            query_transformer = QueryTransformer()
        qm = query_transformer.get_query_metadata(request.text)
        # Optionally derive a small keyword list from entities words
        ents = qm.get("extracted_entities", {}) if isinstance(qm, dict) else {}
        keywords: List[str] = []
        try:
            # Combine some buckets for a simple keyword surface
            for k in ("product_entities", "action_entities", "contact_entities", "time_entities"):
                for v in ents.get(k, []) or []:
                    if isinstance(v, str) and len(v) >= 3:
                        keywords.append(v)
            keywords = sorted(list(set(keywords)))[:10]
        except Exception:
            keywords = []

        return EnhancedNLPResponse(
            language=qm.get("language", "en"),
            primary_intent=qm.get("primary_intent", "general_inquiry"),
            detected_intents=qm.get("detected_intents", []),
            sentiment=qm.get("sentiment", {"label": "neutral", "score": 0.0}),
            priority=qm.get("priority", 3),
            extracted_entities=ents,
            keywords=keywords,
            summary=None
        )
    except Exception as e:
        logger.error(f"Enhanced NLP analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/search/smoke", dependencies=[Depends(require_api_key)])
async def search_smoke_test():
    """Run a quick hybrid + rerank smoke test and return diagnostics."""
    try:
        components = get_components()
        vector_store = components["vector_store"]
        
        sample_queries = [
            "refund policy",
            "shipping information",
            "contact support"
        ]
        results_summary = []
        for q in sample_queries:
            final = vector_store.hybrid_search_with_reranking(q, hybrid_k=10, final_k=3)
            results_summary.append({
                "query": q,
                "final_results_count": len(final),
                "top_snippet": final[0][0].page_content[:120] if final else None
            })
        
        return {
            "success": True,
            "summary": results_summary,
            "hybrid_stats": vector_store.get_hybrid_search_stats()
        }
    except Exception as e:
        logger.error(f"Smoke test failed: {str(e)}")
        return {"success": False, "error": str(e)}

# Initialize system components
async def initialize_system():
    """Initialize all system components."""
    global vector_store_manager, chat_history_manager, llm_response_generator, query_transformer, cross_cultural_negotiator, prescriptive_ai, multilingual_negotiator
    
    try:
        logger.info("Initializing NLP CRM System components...")
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store_manager = VectorStoreManager()
        
        # Set up knowledge base if not already done
        info = vector_store_manager.get_collection_info()
        if info.get("document_count", 0) == 0:
            logger.info("Setting up knowledge base...")
            vector_store_manager.setup_knowledge_base()
        
        # Initialize chat history manager
        logger.info("Initializing chat history manager...")
        chat_history_manager = ChatHistoryManager()
        
        # Initialize LLM response generator
        logger.info("Initializing LLM response generator...")
        llm_response_generator = LLMResponseGenerator(use_mock=not Config.USE_REAL_LLM)
        
        # Initialize query transformer
        logger.info("Initializing query transformer...")
        query_transformer = QueryTransformer()
        
        # Initialize structured product DB
        logger.info("Initializing ProductDB...")
        global product_db
        product_db = ProductDB()
        
        # Initialize cross-cultural negotiator
        logger.info("Initializing Cross-Cultural Negotiator...")
        cross_cultural_negotiator = CrossCulturalNegotiator()
        
        # Initialize prescriptive AI
        logger.info("Initializing Prescriptive AI...")
        prescriptive_ai = PrescriptiveAI()
        
        # Initialize multilingual negotiator
        logger.info("Initializing Multilingual Negotiator...")
        multilingual_negotiator = MultilingualNegotiationEngine()
        
        logger.info("All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Kick off initialization without blocking server startup."""
    try:
        _ensure_ffmpeg_on_path()
        # Run initialization in the background so the server binds immediately
        asyncio.create_task(initialize_system())
        logger.info("Background initialization task scheduled.")
    except Exception as e:
        logger.error(f"Failed to schedule initialization: {str(e)}")

# Whisper ASR serverside transcription (OpenAI Whisper by default)
@app.post("/v1/asr/transcribe", dependencies=[Depends(require_api_key)])
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty audio file")

        # Choose provider: OpenAI Whisper or local whisper model
        if Config.USE_OPENAI_WHISPER:
            try:
                from openai import OpenAI
                client = OpenAI()
                # Write to a temp file with original filename extension
                suffix = ".wav"
                if "." in file.filename:
                    suffix = file.filename[file.filename.rfind("."):]
                tmp_path = None
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp.write(content)
                    tmp.flush()
                    tmp_path = tmp.name
                finally:
                    try:
                        tmp.close()
                    except Exception:
                        pass
                with open(tmp_path, "rb") as audio_fp:
                    tr = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_fp,
                        response_format="json"
                    )
                text = getattr(tr, "text", None) or (tr.get("text") if isinstance(tr, dict) else None)
                if not text:
                    raise RuntimeError("No text in Whisper response")
                return {"success": True, "text": text, "provider": "openai_whisper"}
            except Exception as e:
                logger.error(f"OpenAI Whisper failed: {str(e)}")
                # Fall through to local whisper if available
            finally:
                try:
                    if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception:
                    pass

        # Local whisper fallback (tries openai-whisper, then faster-whisper)
        try:
            text = None
            provider = None
            # Preserve original suffix when possible
            suffix = ".wav"
            if "." in file.filename:
                suffix = file.filename[file.filename.rfind("."):]
            tmp_path = None
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(content)
                tmp.flush()
                tmp_path = tmp.name
            finally:
                try:
                    tmp.close()
                except Exception:
                    pass

            # Attempt with openai-whisper
            try:
                import whisper  # type: ignore
                asr_model = whisper.load_model("base")
                result = asr_model.transcribe(tmp_path)
                text = result.get("text") if isinstance(result, dict) else None
                provider = "local_whisper"
            except Exception as inner_e:
                logger.warning(f"openai-whisper not available or failed: {inner_e}")
                # Attempt with faster-whisper
                try:
                    from faster_whisper import WhisperModel  # type: ignore
                    fw_model = WhisperModel("base")
                    segments, _info = fw_model.transcribe(tmp_path)
                    text_parts = []
                    for seg in segments:
                        try:
                            text_parts.append(getattr(seg, "text", "") or "")
                        except Exception:
                            pass
                    text = " ".join([t.strip() for t in text_parts if t and t.strip()]) or None
                    provider = "faster_whisper"
                except Exception as fw_e:
                    logger.error(f"faster-whisper failed: {fw_e}")
                    raise

            if not text:
                raise RuntimeError("No text produced by local ASR")
            return {"success": True, "text": text, "provider": provider}
        except Exception as e2:
            logger.error(f"Local ASR failed: {str(e2)}")
            raise HTTPException(status_code=500, detail="ASR failed")
        finally:
            try:
                if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ASR transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
