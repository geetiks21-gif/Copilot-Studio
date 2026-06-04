# agent/agent.py  (Groq version)
# LLM:        Groq API (llama-3.1-8b-instant) — fast, free, ~1-3s response
# Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, ~90MB)
# Vector DB:  ChromaDB (local file)
#
# Required env var: GROQ_API_KEY
# Get free key from: console.groq.com

from typing import TypedDict, Annotated, List
import operator
import os
from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()

# Disable LangSmith tracing — removes the 403 error
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ── Groq LLM ─────────────────────────────────
from langchain_groq import ChatGroq

# ── Local embeddings via sentence-transformers ─
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ── Vector store ──────────────────────────────
from langchain_chroma import Chroma

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ─────────────────────────────────────────────
# CONFIG
# Model options (all free on Groq):
#   "llama-3.1-8b-instant"     → fastest (~1s)  ← recommended
#   "llama-3.3-70b-versatile"  → best quality (~3s)
#   "mixtral-8x7b-32768"       → great for technical content (~2s)
# ─────────────────────────────────────────────

GROQ_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ─────────────────────────────────────────────
# 1. TYPED STATE
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    retrieved_docs: List[dict]
    llm_response: str
    final_answer: str
    sources: List[str]
    messages: Annotated[List[BaseMessage], operator.add]
    retrieval_count: int


# ─────────────────────────────────────────────
# 2. SHARED RESOURCES
# ─────────────────────────────────────────────

# Cache embeddings — load once, reuse forever
_embeddings = None

def get_embeddings() -> SentenceTransformerEmbeddings:
    global _embeddings
    if _embeddings is None:
        print("Loading embedding model (one time only)...")
        _embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings

# Cache vectorstore too
_vectorstore = None

def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory="./chroma_db",
            collection_name="equipment_docs",
            embedding_function=get_embeddings()
        )
    return _vectorstore


def get_llm() -> ChatGroq:
    """
    Groq LLM — cloud inference, ~1-3 second response time.
    Free tier: 14,400 requests/day, 30 requests/minute.
    Requires GROQ_API_KEY in environment or .env file.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Set it in your .env file or environment variables. "
            "Get a free key from console.groq.com"
        )
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=1024,
        api_key=api_key
    )


# ─────────────────────────────────────────────
# 3. NODE 1 — RETRIEVAL
# ─────────────────────────────────────────────

def retrieval_node(state: AgentState) -> dict:
    print(f"\n[Node: RETRIEVAL] Query: '{state['query']}'")

    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(
        query=state["query"],
        k=3   # 3 chunks is enough and keeps context short for speed
    )

    retrieved_docs = []
    for doc, score in results:
        retrieved_docs.append({
            "content": doc.page_content,
            "source":  doc.metadata.get("source", "Unknown"),
            "page":    doc.metadata.get("page", ""),
            "score":   round(score, 4)
        })
        print(f"  ✓ {doc.metadata.get('source', 'Unknown')} (score: {score:.4f})")

    return {
        "retrieved_docs":  retrieved_docs,
        "retrieval_count": len(retrieved_docs)
    }


# ─────────────────────────────────────────────
# 4. NODE 2 — REASONING
# ─────────────────────────────────────────────

REASONING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert industrial equipment analyst with deep \
knowledge of mechanical systems, failure modes, and maintenance procedures.

Answer the engineer's question using ONLY the context documents provided.
Be precise, technical, and cite the document name for each key claim.

Rules:
- If context is insufficient, say so clearly — never fabricate data
- Use technical terminology appropriate for engineers
- Lead with the key finding, then supporting detail
- Keep answer concise and focused

Context documents:
{context}
"""),
    ("human", "Question: {query}\n\nAnswer:")
])


def reasoning_node(state: AgentState) -> dict:
    print(f"\n[Node: REASONING] Sending {state['retrieval_count']} docs to Groq ({GROQ_MODEL})")

    context_parts = []
    for i, doc in enumerate(state["retrieved_docs"], 1):
        context_parts.append(
            f"[Doc {i} | {doc['source']}]\n{doc['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    llm = get_llm()
    chain = REASONING_PROMPT | llm
    response = chain.invoke({"context": context, "query": state["query"]})

    print(f"  ✓ Groq response ({len(response.content)} chars)")

    return {
        "llm_response": response.content,
        "messages": [
            HumanMessage(content=state["query"]),
            AIMessage(content=response.content)
        ]
    }


# ─────────────────────────────────────────────
# 5. NODE 3 — FORMATTING
# ─────────────────────────────────────────────

def formatting_node(state: AgentState) -> dict:
    print(f"\n[Node: FORMATTING] Structuring final response")

    seen = set()
    unique_sources = []
    for doc in state["retrieved_docs"]:
        src = doc["source"]
        if src not in seen:
            seen.add(src)
            unique_sources.append(src)

    source_block = "\n\n**Sources:**\n" + "\n".join(
        f"- {src}" for src in unique_sources
    )
    final_answer = state["llm_response"] + source_block

    print(f"  ✓ Done. Sources: {unique_sources}")

    return {
        "final_answer": final_answer,
        "sources":      unique_sources
    }


# ─────────────────────────────────────────────
# 6. BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("retrieval",  retrieval_node)
    graph.add_node("reasoning",  reasoning_node)
    graph.add_node("formatting", formatting_node)

    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval",  "reasoning")
    graph.add_edge("reasoning",  "formatting")
    graph.add_edge("formatting", END)

    return graph.compile(checkpointer=MemorySaver())


# Compile once at module load
agent = build_agent()


# ─────────────────────────────────────────────
# 7. PUBLIC INTERFACE  (called by api.py)
# ─────────────────────────────────────────────

def run_agent(query: str, thread_id: str = "default") -> dict:
    """
    Run the agent for a query.

    Args:
        query:     Engineer's natural language question
        thread_id: Session ID for multi-turn memory

    Returns:
        { "answer": str, "sources": List[str], "retrieval_count": int }
    """
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "query":           query,
        "retrieved_docs":  [],
        "llm_response":    "",
        "final_answer":    "",
        "sources":         [],
        "messages":        [],
        "retrieval_count": 0
    }

    result = agent.invoke(initial_state, config=config)

    return {
        "answer":          result["final_answer"],
        "sources":         result["sources"],
        "retrieval_count": result["retrieval_count"]
    }


# ─────────────────────────────────────────────
# 8. LOCAL TEST
#    python agent.py "your question here"
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    test_query = (
        sys.argv[1] if len(sys.argv) > 1
        else "What caused the P-101 incidents in January 2024?"
    )

    print("=" * 60)
    print(f"MODEL:  {GROQ_MODEL}  (Groq cloud)")
    print(f"EMBED:  {EMBEDDING_MODEL}  (local)")
    print(f"QUERY:  {test_query}")
    print("=" * 60)

    start = time.time()
    response = run_agent(query=test_query, thread_id="test-1")
    elapsed = round(time.time() - start, 2)

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(response["answer"])
    print(f"\nSources:      {response['sources']}")
    print(f"Docs used:    {response['retrieval_count']}")
    print(f"Total time:   {elapsed}s")