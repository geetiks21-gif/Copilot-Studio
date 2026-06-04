# api.py
# FastAPI wrapper for the LangGraph agent
# Exposes POST /query endpoint that Copilot Studio calls
#
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import traceback

# Import your agent (must be in same folder)
from agent import run_agent


# ─────────────────────────────────────────────
# 1. APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Equipment Q&A Agent",
    description="LangGraph RAG agent for industrial equipment queries. "
                "Powered by local Ollama + ChromaDB.",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc"
)

# CORS — required for Copilot Studio and browser-based clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# 2. REQUEST / RESPONSE MODELS
#    Pydantic models define the exact JSON
#    shape Copilot Studio sends and receives
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str                       # the engineer's query
    session_id: Optional[str] = "default"  # for multi-turn memory

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What caused the P-101 incidents in January 2024?",
                "session_id": "user-123"
            }
        }


class QueryResponse(BaseModel):
    answer: str                         # full answer with sources
    sources: List[str]                  # list of source documents cited
    retrieval_count: int                # how many chunks were used
    session_id: str                     # echoed back for client tracking

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Pump P-101 had three incidents...",
                "sources": ["pump_p101_incident_log.txt"],
                "retrieval_count": 5,
                "session_id": "user-123"
            }
        }


class HealthResponse(BaseModel):
    status: str
    model: str
    vector_store: str
    message: str


# ─────────────────────────────────────────────
# 3. ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    """API info — useful to confirm ngrok tunnel is working."""
    return {
        "name": "Equipment Q&A Agent",
        "version": "1.0.0",
        "endpoints": {
            "query":  "POST /query",
            "health": "GET  /health",
            "docs":   "GET  /docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Health check endpoint.
    Copilot Studio plugin registration pings this to verify the API is live.
    Also useful for monitoring.
    """
    try:
        # Try importing ChromaDB to verify vector store is accessible
        from langchain_chroma import Chroma
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vs = Chroma(
            persist_directory="./chroma_db",
            collection_name="equipment_docs",
            embedding_function=embeddings
        )
        doc_count = vs._collection.count()
        vector_store_status = f"OK ({doc_count} vectors)"
    except Exception as e:
        vector_store_status = f"ERROR: {str(e)}"

    return HealthResponse(
        status="ok",
        model="llama-3.1-8b-instant (Groq)",
        vector_store=vector_store_status,
        message="Agent is ready to receive queries"
    )


@app.post("/query", response_model=QueryResponse, tags=["Agent"])
async def query_agent(request: QueryRequest):
    """
    Main endpoint — receives a question, returns the agent's answer.

    This is the endpoint Copilot Studio Plugin Action calls.

    Request body:
        question:   The engineer's natural language question
        session_id: Optional session ID for multi-turn memory

    Returns:
        answer:           Full answer text with sources appended
        sources:          List of source document names cited
        retrieval_count:  Number of document chunks retrieved
        session_id:       Echoed back for client tracking
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = run_agent(
            query=request.question,
            thread_id=request.session_id
        )

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            retrieval_count=result["retrieval_count"],
            session_id=request.session_id
        )

    except Exception as e:
        # Log full traceback for debugging
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )


# ─────────────────────────────────────────────
# 4. OPENAPI SCHEMA ENDPOINT
#    Copilot Studio reads this to understand
#    your API's structure when you register it
#    as a plugin action
# ─────────────────────────────────────────────

@app.get("/openapi-plugin.json", tags=["Plugin"])
def openapi_plugin():
    """
    Returns plugin manifest for Copilot Studio registration.
    Copilot Studio reads this URL when you add a plugin action.
    """
    return {
        "schema_version": "v1",
        "name_for_human": "Equipment Q&A Agent",
        "name_for_model": "equipment_agent",
        "description_for_human": "Ask questions about industrial equipment, "
                                  "failure modes, incidents, and maintenance.",
        "description_for_model": "Use this plugin to answer questions about "
                                  "industrial equipment including pump failures, "
                                  "vibration analysis, maintenance schedules, "
                                  "and incident history.",
        "contact_email": "admin@yourcompany.com",
        "api": {
            "type": "openapi",
            "url": "/openapi.json"   # FastAPI auto-generates this
        }
    }


# ─────────────────────────────────────────────
# 5. RUN DIRECTLY
#    python api.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True       # auto-restarts on code changes
    )