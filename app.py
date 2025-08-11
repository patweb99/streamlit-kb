# ===========================================================
# 🧠 GenAI Knowledgebase (Strands + Bedrock + ChromaDB)
# ===========================================================
# 📚 Classroom Objective:
# This application demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline:
#  1) Upload & process documents (PDF, Markdown, TXT)
#  2) Split text into chunks
#  3) Create vector embeddings (Amazon Titan via AWS Bedrock, using boto3)
#  4) Store vectors in ChromaDB (local persistent vector database)
#  5) Retrieve the most relevant chunks using vector similarity
#  6) Answer questions with an LLM (via Strands Agent + Bedrock), grounded in retrieved chunks
#
# 🔧 Tech choices:
#  - No LangChain: embedding and vector store are called directly
#  - Strands Agents: small agent framework to define LLM + tool(s)
#  - ChromaDB: local vector database with persistent storage
#  - Streamlit: simple web UI for uploads, indexing, debugging, and Q&A
#
# -----------------------------------------------------------
# REQUIREMENTS (install once):
#   pip install streamlit python-dotenv PyPDF2 boto3 chromadb strands-agents
#
# -----------------------------------------------------------
# AWS ENVIRONMENT:
# You must have AWS credentials that allow Bedrock model invocation.
# Minimum permission: bedrock:InvokeModel
#
# Optional .env file in the same directory:
#   AWS_REGION=us-west-2
#   EMBED_MODEL_ID=amazon.titan-embed-text-v1
#   LLM_MODEL_ID=us.anthropic.claude-3-5-haiku-20241022-v1:0
#
# -----------------------------------------------------------
# HOW TO RUN:
#   streamlit run app.py
#
# -----------------------------------------------------------
# TEACHING TIPS:
# - Start by uploading a small .txt or .md file (or a short PDF).
# - Click "Re-index Knowledgebase" so embeddings are created & stored.
# - Use "Test Retrieval" to verify that chunks are being found.
# - Then ask a question in "Ask via LLM" and inspect the cited chunks.
# - Emphasize that tools must not call Streamlit APIs (they can run off-thread).
# ===========================================================

import os
import json
import time
import tempfile
import threading
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
import PyPDF2

import boto3
from botocore.config import Config

import chromadb
from chromadb.config import Settings

# Strands Agents — a tiny agent framework for LLM + tools
from strands import Agent, tool
from strands.models import BedrockModel


# ===========================================================
# 🌎 Environment & App Configuration
# ===========================================================
st.set_page_config(page_title="🧠 Knowledgebase (Strands + Bedrock)", layout="wide")

# Load variables from .env if present (useful for AWS_REGION, etc.)
load_dotenv()

# Region & model IDs: you can override via .env or edit defaults here
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v1")  # Titan embeddings
DEFAULT_LLM_MODEL_ID = os.getenv(
    "LLM_MODEL_ID",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0"  # Bedrock Anthropic model (Claude Haiku as example)
)

# Folder to store uploads & converted text
DATA_DIR = "data"
# Temporary root folder to store Chroma persistent data for this run
TEMP_DIR = tempfile.mkdtemp(prefix="kb_chroma_")
PERSIST_ROOT = os.path.join(TEMP_DIR, "chroma_db")

# Ensure the data directory exists so uploads don’t fail
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------
# GLOBALS (module-level, because tools must not call streamlit)
# -----------------------------------------------------------
CHROMA_CLIENT = None       # Chroma persistent client (reopened on rerun)
COLLECTION = None          # Chroma collection handle (reopened on rerun)
COLLECTION_NAME = "kb_main"  # Must be 3–512 chars; allowed [a-zA-Z0-9._-]

# Retrieval settings (default top-K)
K_RETRIEVE = 3

# Thread-safe buffer to store last retrieved sources so the UI can render them
_LAST_SOURCES: List[Dict[str, Any]] = []
_LAST_SOURCES_LOCK = threading.Lock()


def _set_last_sources(items: List[Dict[str, Any]]) -> None:
    """
    Store last retrieved chunks in a thread-safe way.
    Tools should not call Streamlit APIs, so we use this buffer to pass data back to the UI.
    """
    global _LAST_SOURCES
    with _LAST_SOURCES_LOCK:
        _LAST_SOURCES = items


def _get_last_sources() -> List[Dict[str, Any]]:
    """Access the last retrieved chunks (read-only copy)."""
    with _LAST_SOURCES_LOCK:
        return list(_LAST_SOURCES)


# ===========================================================
# 🔌 AWS Bedrock — Client & Embedding Function
# ===========================================================
def bedrock_client():
    """
    Bedrock Runtime client — used for invoking embeddings or text models.
    The account/role you run under must have bedrock:InvokeModel permissions.
    """
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        config=Config(retries={"max_attempts": 5, "mode": "standard"})
    )


def titan_embed(text: str) -> List[float]:
    """
    Create a vector embedding using Amazon Titan (Text Embeddings) via Bedrock.
    - Embeddings are numeric vectors that represent text in a high-dimensional space.
    - Similar texts → similar vectors → small distance in vector space.

    Returns:
        list[float]: The embedding vector.
    """
    if not text or not text.strip():
        return []

    # Titan v1 expects {"inputText": "..."} as the body
    body = {"inputText": text}

    resp = bedrock_client().invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read())
    return payload.get("embedding", [])


# ===========================================================
# 📄 PDF & Text Utilities (No LangChain)
# ===========================================================
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract plain text from a PDF file using PyPDF2.

    NOTE (teaching):
    - PDF text extraction is best-effort; complex layout/graphics can degrade results.
    - For best RAG quality, prefer clean .txt or .md source files when available.
    """
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text() or ""
            if t:
                text += t + "\n\n"
    return text.strip()


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Naive character-based text splitter with overlap.
    Teaching notes:
    - Overlap preserves context continuity across chunk boundaries.
    - More sophisticated splitters (by sentence/paragraph) can improve retrieval precision.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


# ===========================================================
# 💾 ChromaDB Helpers (Native Client, Persistent)
# ===========================================================
def new_chroma_client(persist_dir: str):
    """
    Create a Chroma persistent client rooted at `persist_dir`.
    Teaching note: Chroma can run in-memory or persist to disk; we choose persistence so
    state survives Streamlit reruns.
    """
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def reset_collection(client, name: str):
    """
    Create a clean collection with the given name, deleting any existing one.
    Teaching note: In production, you might want versioning instead of delete/recreate.
    """
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        client.delete_collection(name)
    return client.create_collection(name=name)


def load_collection_from_persist() -> bool:
    """
    Rehydrate the persistent Chroma client/collection after Streamlit reruns.
    Teaching note: Streamlit reruns the script on every interaction, so we reopen the collection
    from a path stored in session_state, if available.
    """
    global CHROMA_CLIENT, COLLECTION
    persist_dir = st.session_state.get("persist_dir")
    if not persist_dir:
        return False
    try:
        CHROMA_CLIENT = new_chroma_client(persist_dir)
        COLLECTION = CHROMA_CLIENT.get_or_create_collection(COLLECTION_NAME)
        return True
    except Exception as e:
        st.error(f"Failed to load collection: {e}")
        return False


# ===========================================================
# 🧱 Indexing Pipeline
# ===========================================================
def reindex_knowledgebase() -> bool:
    """
    1) Reads all .txt/.md files in DATA_DIR
    2) Splits into overlapping chunks
    3) Creates embeddings with Titan
    4) Stores documents + embeddings + metadata in Chroma
    """
    global CHROMA_CLIENT, COLLECTION

    # 1) Gather file contents
    docs, ids, metadatas = [], [], []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]
    if not files:
        st.error("No .txt or .md files found. Upload or convert PDFs first.")
        return False

    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            st.warning(f"Skipping {fname}: {e}")
            continue

        if not content.strip():
            st.warning(f"Skipping empty file: {fname}")
            continue

        # 2) Split into chunks
        chunks = split_text(content, chunk_size=500, overlap=50)
        for i, ch in enumerate(chunks):
            docs.append(ch)
            ids.append(f"{fname}-{i}")                # Unique ID per chunk
            metadatas.append({"source": fname, "chunk": i})  # Store source & chunk index

    if not docs:
        st.error("No valid chunks produced. Check your files.")
        return False

    # 3) Create a new persistent directory for this index run
    persist_dir = os.path.join(PERSIST_ROOT, f"run_{int(time.time())}")
    os.makedirs(persist_dir, exist_ok=True)

    CHROMA_CLIENT = new_chroma_client(persist_dir)
    COLLECTION = reset_collection(CHROMA_CLIENT, COLLECTION_NAME)

    # 4) Embed and add to Chroma (simple per-item loop; easy to follow)
    batch_size = 32
    total = len(docs)
    embeddings = []

    with st.spinner("Embedding chunks with Titan..."):
        for i in range(0, total, batch_size):
            batch = docs[i : i + batch_size]
            for b in batch:
                embeddings.append(titan_embed(b))
            st.write(f"Embedded {min(i + batch_size, total)}/{total} chunks")

    # Add to Chroma (documents + embeddings + metadata + ids)
    COLLECTION.add(documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids)

    # Save persistence path in session so we can reopen after rerun
    st.session_state["vectorstore_loaded"] = True
    st.session_state["persist_dir"] = persist_dir

    st.success(f"✅ Indexed {len(docs)} chunks from {len(set(m['source'] for m in metadatas))} file(s).")
    try:
        st.info(f"🔢 Collection count: {COLLECTION.count()}")
    except Exception:
        pass
    return True


# ===========================================================
# 🛠️ Retrieval Tool (Strands) — NO Streamlit calls inside!
# ===========================================================
@tool
def retrieve_chunks(question: str) -> str:
    """
    Retrieval tool for the Strands Agent.

    Contract:
    - Uses the global Chroma COLLECTION (already built/loaded).
    - Embeds the user question with Titan.
    - Queries top-K similar chunks from Chroma.
    - Stores the results in a thread-safe buffer for the UI to display.
    - Returns a formatted context block for the agent to read.

    IMPORTANT:
    - This function must not call any Streamlit APIs (can run off the main thread).
    - Use the _set_last_sources() helper to pass data back to the UI.
    """
    global COLLECTION, K_RETRIEVE
    if COLLECTION is None:
        return "[No index loaded]"

    # 1) Embed the question
    q_vec = titan_embed(question)

    # 2) Query Chroma for similar chunks
    #    include=["documents","metadatas","distances"] → ids are returned by default in current Chroma versions,
    #    but are NOT valid values in include[] for some versions. We avoid "ids" in include to be safe.
    res = COLLECTION.query(
        query_embeddings=[q_vec],
        n_results=max(1, int(K_RETRIEVE)),
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    # 3) Store for UI (thread-safe)
    packed = []
    for d, m, dist in zip(docs, metas, dists):
        packed.append(
            {
                "text": d or "",
                "meta": m or {},
                "distance": float(dist) if dist is not None else None,
            }
        )
    _set_last_sources(packed)

    # 4) Build an agent-readable context block
    out_lines = []
    for rank, item in enumerate(packed, start=1):
        src = item["meta"].get("source", "unknown")
        dist = item.get("distance")
        dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else "NA"
        out_lines.append(f"[Source {rank} — {src} — dist:{dist_str}]\n{item['text']}")
    return "\n\n".join(out_lines) if out_lines else "[No matching context]"


# ===========================================================
# 🖥️ Streamlit UI
# ===========================================================
st.title("🧠 Knowledgebase (Strands + Bedrock) — No LangChain")

# Track whether we’ve created an index during this session
if "vectorstore_loaded" not in st.session_state:
    st.session_state["vectorstore_loaded"] = False

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📤 Upload & Re-Index", "🗑️ Delete Files", "💬 Ask Questions"],
    index=0,
)

# -----------------------------------------------------------
# PAGE: Upload & Re-Index
# -----------------------------------------------------------
if page == "📤 Upload & Re-Index":
    st.header("📤 Upload Files")
    st.info(
        "Upload .pdf/.md/.txt files. After uploading, click **Re-index Knowledgebase** to build the vector index."
    )

    # Upload widget
    uploaded = st.file_uploader(
        "Upload .pdf, .md, .txt", type=["pdf", "md", "txt"], accept_multiple_files=True
    )
    if uploaded:
        for up in uploaded:
            original = up.name
            dest_path = os.path.join(DATA_DIR, original)

            try:
                if original.lower().endswith(".pdf"):
                    # Teaching note: Convert PDF → .txt so we can index plain text.
                    pdf_tmp = dest_path + ".tmp"
                    with open(pdf_tmp, "wb") as f:
                        f.write(up.read())
                    text = extract_text_from_pdf(pdf_tmp)
                    os.remove(pdf_tmp)

                    if text:
                        txt_path = os.path.join(DATA_DIR, original[:-4] + ".txt")
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(text)
                        st.success(f"Converted {original} → {os.path.basename(txt_path)}")
                    else:
                        st.error(f"Failed to extract text from {original}")
                else:
                    # Save .txt / .md directly
                    with open(dest_path, "wb") as f:
                        f.write(up.read())
                    st.success(f"Saved {original}")
            except Exception as e:
                st.error(f"Error handling {original}: {e}")

    # Build the index
    if st.button("🔄 Re-index Knowledgebase", type="primary"):
        ok = reindex_knowledgebase()
        if not ok:
            st.warning("Indexing failed or no documents found.")

    # File listing
    existing = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]
    if existing:
        st.subheader("📋 Current Files")
        for f in existing:
            st.write("•", f)


# -----------------------------------------------------------
# PAGE: Delete Files
# -----------------------------------------------------------
elif page == "🗑️ Delete Files":
    st.header("🗑️ Delete Files")
    files = [f for f in os.listdir(DATA_DIR) if f.endswith((".txt", ".md"))]
    selected = st.multiselect("Select files to delete", files)
    if st.button("Delete Selected"):
        deleted = 0
        for f in selected:
            try:
                os.remove(os.path.join(DATA_DIR, f))
                deleted += 1
            except Exception as e:
                st.error(f"Error deleting {f}: {e}")
        if deleted:
            st.success(f"Deleted {deleted} file(s). Re-index to refresh the database.")


# -----------------------------------------------------------
# PAGE: Ask Questions
# -----------------------------------------------------------
elif page == "💬 Ask Questions":
    st.header("💬 Ask Questions")

    # Streamlit reruns can drop globals — rehydrate Chroma from disk if needed
    if COLLECTION is None:
        load_collection_from_persist()

    # Quick health check (count chunks)
    if COLLECTION is not None:
        try:
            st.info(f"📦 Collection: {COLLECTION_NAME} | 🔢 Chunks: {COLLECTION.count()}")
        except Exception as e:
            st.warning(f"Could not read collection count: {e}")

    if not st.session_state.get("vectorstore_loaded", False) and COLLECTION is None:
        st.warning("⚠️ No index loaded. Go to **Upload & Re-Index** and build the index first.")
    else:
        # ---- Retrieval Settings (controls the tool's top-K)
        st.subheader("🔎 Retrieval Settings")
        k_value = st.slider("Top-K chunks to retrieve", 1, 10, 3, 1)
        # Assigning at the top level of the script updates the module variable (no 'global' needed here)
        K_RETRIEVE = int(k_value)

        # ---- Debug path (bypass LLM) to verify chunks are available
        st.subheader("🧪 Test Retrieval (No LLM)")
        debug_q = st.text_input("Test query", value="test")
        if st.button("Run Test Retrieval"):
            if COLLECTION is None:
                st.error("No index loaded.")
            else:
                q_vec = titan_embed(debug_q)
                res = COLLECTION.query(
                    query_embeddings=[q_vec],
                    n_results=K_RETRIEVE,
                    include=["documents", "metadatas", "distances"],  # do not include "ids" here
                )
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]

                if not docs:
                    st.warning("No chunks returned. Try a different query or re-index.")
                else:
                    # Show a quick table preview of retrieved chunks
                    import pandas as pd
                    table = pd.DataFrame([
                        {
                            "Rank": i + 1,
                            "Source": (metas[i] or {}).get("source", "unknown"),
                            "Chunk#": (metas[i] or {}).get("chunk", None),
                            "Distance": float(dists[i]) if dists and dists[i] is not None else None,
                            "Preview": (docs[i] or "")[:140].replace("\n", " ") + ("…" if len(docs[i]) > 140 else "")
                        }
                        for i in range(len(docs))
                    ])
                    st.dataframe(table, use_container_width=True)

                    # Full text expanders
                    for i, d in enumerate(docs, start=1):
                        src = (metas[i-1] or {}).get("source", "unknown")
                        chk = (metas[i-1] or {}).get("chunk", None)
                        dist = float(dists[i-1]) if dists and dists[i-1] is not None else None
                        label = f"Result {i} — {src} — chunk {chk}" + (f" — distance {dist:.4f}" if dist is not None else "")
                        with st.expander(label):
                            st.write(d)

        # ---- Ask via LLM (Strands Agent calls the retrieval tool first)
        st.subheader("💬 Ask via LLM")
        model_id = st.text_input("Bedrock model ID", value=DEFAULT_LLM_MODEL_ID, help="e.g., us.amazon.nova-micro-v1:0")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

        # Build the Bedrock-backed Strands model + agent
        bedrock_model = BedrockModel(model_id=model_id, temperature=temperature, region=AWS_REGION)
        agent = Agent(model=bedrock_model, tools=[retrieve_chunks])

        question = st.text_input("Your question")
        if st.button("Generate Answer") and question:
            with st.spinner("Retrieving and generating answer..."):
                system_preamble = (
                    "You are a helpful assistant that answers using local documents. "
                    "First call the `retrieve_chunks` tool with the user's question, then answer concisely. "
                    "If you use any context, cite it inline as [Source 1], [Source 2], etc. "
                    "If no context is relevant, say so."
                )
                response = agent(f"{system_preamble}\n\nUser question: {question}")

                # Show final answer
                st.markdown("### 💬 Answer")
                st.write(str(response))

                # Show retrieved chunks (the exact ones the tool pulled)
                st.markdown("### 📄 Retrieved Chunks (from Chroma)")
                rows = _get_last_sources()
                if not rows:
                    st.info("No chunks returned.")
                else:
                    import pandas as pd
                    table = pd.DataFrame([
                        {
                            "Rank": i + 1,
                            "Source": r["meta"].get("source", "unknown"),
                            "Chunk#": r["meta"].get("chunk", None),
                            "Distance": r.get("distance", None),
                            "Preview": (r["text"] or "")[:140].replace("\n", " ") + ("…" if len(r["text"]) > 140 else "")
                        }
                        for i, r in enumerate(rows)
                    ])
                    st.dataframe(table, use_container_width=True)

                    # Expanders with full text for transparency
                    for i, r in enumerate(rows, start=1):
                        src = r["meta"].get("source", "unknown")
                        chk = r["meta"].get("chunk", None)
                        dist = r.get("distance", None)
                        title = f"Source {i} — {src} — chunk {chk}" + (f" — distance {dist:.4f}" if isinstance(dist, (int, float)) else "")
                        with st.expander(title):
                            st.write(r["text"] or "")
