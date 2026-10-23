"""
Streamlit Web Application for Hybrid Search RAG Pipeline.
==========================================================
Features:
- Conversational / Q&A interface with grounded answers and source citations
- Upload and ingest custom PDFs, TXT, MD, and HTML documents into the pipeline
- Filter context retrieval to specific uploaded documents or search across all documents
- Real-time connection to FastAPI backend (http://localhost:8000)
- Advanced retrieval controls (Dense + BM25 Hybrid vs Dense-only, Top-K, Confidence threshold)
- Interactive citation inspection (Rerank scores, source document, section headings)
- Knowledge base explorer showing indexed documents and chunk statistics
"""

import streamlit as st
import requests
import pandas as pd
from typing import Dict, Any, List, Optional

# =========================================================================
# Page Configuration & Styling
# =========================================================================
st.set_page_config(
    page_title="Hybrid RAG Search Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Main container styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-chip {
        display: inline-flex;
        align-items: center;
        background-color: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 9999px;
        padding: 2px 10px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border-color: rgba(16, 185, 129, 0.3);
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border-color: rgba(245, 158, 11, 0.3);
    }
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border-color: rgba(239, 68, 68, 0.3);
    }
    .chunk-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .chunk-title {
        font-weight: 600;
        color: #E2E8F0;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    .chunk-meta {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .chunk-text {
        font-size: 0.88rem;
        color: #CBD5E1;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    .filter-banner {
        padding: 8px 14px;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        font-size: 0.9rem;
        color: #93C5FD;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# Backend API Client Helpers
# =========================================================================

def get_backend_health(base_url: str) -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(f"{base_url}/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_indexed_documents(base_url: str) -> List[Dict[str, Any]]:
    try:
        resp = requests.get(f"{base_url}/v1/documents", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def get_strategies(base_url: str) -> List[str]:
    try:
        resp = requests.get(f"{base_url}/v1/strategies", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def query_rag_api(
    base_url: str,
    question: str,
    use_sparse: bool,
    top_k: int,
    retrieval_threshold: float,
    filter_document: Optional[str] = None,
    strategy: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "question": question,
        "use_sparse": use_sparse,
        "top_k": top_k,
        "retrieval_threshold": retrieval_threshold,
    }
    if filter_document and filter_document != "All Documents":
        payload["filter_document"] = filter_document
    if strategy and strategy != "All":
        payload["strategy"] = strategy

    resp = requests.post(f"{base_url}/v1/ask", json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()


def upload_document_api(
    base_url: str,
    file_bytes: bytes,
    filename: str,
    strategy: str = "recursive_character",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Dict[str, Any]:
    files = {"file": (filename, file_bytes)}
    data = {
        "strategy": strategy,
        "chunk_size": str(chunk_size),
        "chunk_overlap": str(chunk_overlap),
    }
    resp = requests.post(f"{base_url}/v1/upload", files=files, data=data, timeout=120)
    resp.raise_for_status()
    return resp.json()


def ingest_chunk_api(
    base_url: str,
    text: str,
    source_document: str,
    chunk_index: int = 0,
    section_heading: str = "",
    chunking_strategy: str = "custom",
) -> Dict[str, Any]:
    payload = {
        "chunks": [
            {
                "text": text,
                "source_document": source_document,
                "chunk_index": chunk_index,
                "section_heading": section_heading,
                "chunking_strategy": chunking_strategy,
            }
        ]
    }
    resp = requests.post(f"{base_url}/v1/ingest", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


# =========================================================================
# Sidebar Controls & Backend Status
# =========================================================================

st.sidebar.markdown("## ⚙️ Pipeline Configuration")

backend_url = st.sidebar.text_input(
    "FastAPI Backend URL",
    value="http://127.0.0.1:8000",
    help="Target host and port of the running FastAPI service.",
)

# Health Check
health_data = get_backend_health(backend_url)
if health_data:
    st.sidebar.markdown(
        f"""
        <div style="padding: 8px 12px; background: rgba(16, 185, 129, 0.1); border-radius: 6px; border: 1px solid rgba(16, 185, 129, 0.3); margin-bottom: 12px;">
            <span style="color: #34D399; font-weight: 600;">● Online</span> | <b>{health_data.get('total_chunks_indexed', 0)}</b> chunks indexed
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        """
        <div style="padding: 8px 12px; background: rgba(239, 68, 68, 0.1); border-radius: 6px; border: 1px solid rgba(239, 68, 68, 0.3); margin-bottom: 12px;">
            <span style="color: #F87171; font-weight: 600;">● Disconnected</span>
            <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">
                Make sure the backend is running with: <code>uvicorn backend.app:app --reload</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Retrieval Target Scope")

# Fetch indexed documents list for filtering
indexed_docs = get_indexed_documents(backend_url) if health_data else []
doc_names = [d["source_document"] for d in indexed_docs]
unique_doc_names = sorted(list(set(doc_names)))

doc_filter_options = ["All Documents"] + unique_doc_names

# Allow setting selected document from session state (e.g. after upload)
if "selected_document_filter" not in st.session_state:
    st.session_state.selected_document_filter = "All Documents"

if st.session_state.selected_document_filter not in doc_filter_options:
    st.session_state.selected_document_filter = "All Documents"

selected_doc_index = doc_filter_options.index(st.session_state.selected_document_filter)

selected_document = st.sidebar.selectbox(
    "Target Document Scope",
    options=doc_filter_options,
    index=selected_doc_index,
    help="Restricts context retrieval exclusively to the selected document (e.g. your uploaded PDF) or searches across all documents.",
)
st.session_state.selected_document_filter = selected_document

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Retrieval Parameters")

use_sparse = st.sidebar.toggle(
    "Hybrid Retrieval (Dense + BM25)",
    value=True,
    help="When enabled, fuses dense embeddings with BM25 keyword search via Reciprocal Rank Fusion (RRF), then reranks with a Cross-Encoder.",
)

top_k = st.sidebar.slider(
    "Top-K Reranked Chunks",
    min_value=1,
    max_value=15,
    value=4,
    step=1,
    help="Number of most relevant context chunks to pass to the LLM.",
)

retrieval_threshold = st.sidebar.slider(
    "Hallucination Guardrail Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.05,
    help="Minimum cross-encoder confidence required before answering. Protects against hallucinations when query is out-of-domain.",
)

# Fetch strategies
available_strategies = get_strategies(backend_url) if health_data else []
strategy_options = ["All"] + available_strategies
selected_strategy = st.sidebar.selectbox(
    "Chunking Strategy Filter",
    options=strategy_options,
    index=0,
    help="Filter retrieval to specific chunking strategy or search across all chunks.",
)

if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()


# =========================================================================
# Main Layout Tabs
# =========================================================================

st.markdown('<div class="main-title">⚡ Hybrid Search & Grounded RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Dense vector retrieval (ChromaDB) + Sparse BM25 + Reciprocal Rank Fusion + Local LLM Synthesis</div>',
    unsafe_allow_html=True,
)

tab_chat, tab_upload, tab_docs = st.tabs(["💬 Ask & Retrieve", "📤 Upload Documents", "📚 Knowledge Base Explorer"])


# =========================================================================
# Tab 1: Ask & Retrieve
# =========================================================================
with tab_chat:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Active Filter Banner
    scope_display = "All Indexed Documents" if selected_document == "All Documents" else f"📄 Filtered strictly to: <b>{selected_document}</b>"
    st.markdown(
        f"""
        <div class="filter-banner">
            🎯 <b>Retrieval Scope:</b> {scope_display} | 
            <b>Mode:</b> {'Hybrid (Dense + BM25)' if use_sparse else 'Dense Only'} | 
            <b>Top-K:</b> {top_k} | 
            <b>Guardrail Threshold:</b> {retrieval_threshold}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Example question chips
    st.markdown("**Quick Prompts:**")
    example_cols = st.columns(3)
    sample_queries = [
        "What is the relationship between humans and the environment?",
        "What are the major non-renewable energy resources?",
        "What are key methods for environmental conservation?",
    ]
    
    preset_query = None
    for idx, sample_text in enumerate(sample_queries):
        if example_cols[idx].button(f"💡 {sample_text[:40]}...", key=f"sample_{idx}", use_container_width=True):
            preset_query = sample_text

    # Render previous conversation history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("status") == "insufficient_context":
                st.warning("⚠️ Retrieval score fell below safety threshold. Answer was suppressed to prevent hallucination.")
            if message.get("filter_used") and message["filter_used"] != "All Documents":
                st.caption(f"🎯 Context scoped to: {message['filter_used']}")
            if message.get("sources"):
                with st.expander(f"📑 View {len(message['sources'])} Cited Source Passages", expanded=False):
                    for src in message["sources"]:
                        score_text = f"Score: {src['score']:.3f}" if src.get("score") is not None else ""
                        st.markdown(
                            f"""
                            <div class="chunk-card">
                                <div class="chunk-title">[{src.get('n', '#')}] {src.get('source_document', 'Unknown')} {f"- {src['section_heading']}" if src.get('section_heading') else ""}</div>
                                <div class="chunk-meta">ID: <code>{src.get('id', '')}</code> | {score_text}</div>
                                <div class="chunk-text">{src.get('text', '')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Chat Input
    user_prompt = st.chat_input("Ask a question grounded in your documents...")
    if preset_query and not user_prompt:
        user_prompt = preset_query

    if user_prompt:
        # Display user message
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Process via backend
        with st.chat_message("assistant"):
            if not health_data:
                err_msg = "❌ Cannot connect to the FastAPI backend. Please check that Uvicorn is running on the configured URL."
                st.error(err_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": err_msg})
            else:
                with st.spinner(f"Retrieving context from {selected_document} and generating answer..."):
                    try:
                        result = query_rag_api(
                            base_url=backend_url,
                            question=user_prompt,
                            use_sparse=use_sparse,
                            top_k=top_k,
                            retrieval_threshold=retrieval_threshold,
                            filter_document=selected_document,
                            strategy=selected_strategy,
                        )

                        status = result.get("status", "unknown")
                        answer_text = result.get("answer", "")
                        sources = result.get("sources", [])
                        confidence = result.get("confidence", {}).get("retrieval_confidence", 0.0)

                        # Header metrics
                        metric_cols = st.columns([2, 2, 2, 4])
                        metric_cols[0].metric("Status", status.title())
                        metric_cols[1].metric("Confidence", f"{confidence:.3f}")
                        metric_cols[2].metric("Sources Used", len(sources))
                        metric_cols[3].caption(f"Context Scope: **{selected_document}**")

                        if status == "insufficient_context":
                            st.warning(
                                f"⚠️ **Low Retrieval Confidence ({confidence:.3f} < {retrieval_threshold:.3f})**\n\n"
                                f"{answer_text or 'The system did not find sufficiently confident evidence in the specified documents to answer this question accurately.'}"
                            )
                        else:
                            st.markdown(answer_text)

                        # Citations display
                        if sources:
                            with st.expander(f"📑 Inspect {len(sources)} Retrieved Sources & Relevance Scores", expanded=False):
                                for src in sources:
                                    score_val = src.get("score")
                                    score_str = f"Rerank Score: <b>{score_val:.4f}</b>" if score_val is not None else ""
                                    st.markdown(
                                        f"""
                                        <div class="chunk-card">
                                            <div class="chunk-title">[{src['n']}] {src['source_document']} {f"- {src['section_heading']}" if src.get('section_heading') else ""}</div>
                                            <div class="chunk-meta">Chunk ID: <code>{src['id']}</code> | {score_str}</div>
                                            <div class="chunk-text">{src.get('text', '')}</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                        # Save to history
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": answer_text or "(No answer generated)",
                                "status": status,
                                "sources": sources,
                                "confidence": confidence,
                                "filter_used": selected_document,
                            }
                        )

                    except requests.exceptions.RequestException as e:
                        err_text = f"⚠️ API Request failed: {str(e)}"
                        st.error(err_text)
                        st.session_state.chat_history.append({"role": "assistant", "content": err_text})


# =========================================================================
# Tab 2: Upload Documents & Custom Ingestion
# =========================================================================
with tab_upload:
    st.subheader("📤 Upload Documents into RAG Pipeline")
    st.caption("Upload your PDFs, TXT, MD, or HTML documents. They are automatically chunked, embedded into ChromaDB, and indexed in BM25 for hybrid search.")

    upload_col1, upload_col2 = st.columns([3, 2])

    with upload_col1:
        st.markdown("#### 1. Select Document File")
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=["pdf", "txt", "md", "html", "htm"],
            help="Supported formats: PDF, Text (.txt), Markdown (.md), HTML (.html)",
        )

    with upload_col2:
        st.markdown("#### 2. Chunking Settings")
        chunk_strat = st.selectbox(
            "Text Splitting Strategy",
            options=["recursive_character", "character", "token"],
            index=0,
            help="recursive_character keeps paragraphs and sentences intact; character splits on newlines; token splits on token boundaries.",
        )
        c_size = st.number_input("Chunk Size (characters/tokens)", min_value=200, max_value=3000, value=1000, step=100)
        c_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)

    if uploaded_file is not None:
        st.markdown(f"**Selected File:** `{uploaded_file.name}` ({uploaded_file.size / 1024:.1f} KB)")
        
        process_btn = st.button("🚀 Process & Ingest Document", use_container_width=True, type="primary")

        if process_btn:
            if not health_data:
                st.error("Cannot connect to backend. Please ensure Uvicorn backend is running.")
            else:
                with st.spinner(f"Extracting, chunking, and embedding `{uploaded_file.name}` into ChromaDB & BM25..."):
                    try:
                        file_bytes = uploaded_file.getvalue()
                        res = upload_document_api(
                            base_url=backend_url,
                            file_bytes=file_bytes,
                            filename=uploaded_file.name,
                            strategy=chunk_strat,
                            chunk_size=int(c_size),
                            chunk_overlap=int(c_overlap),
                        )

                        st.success(
                            f"🎉 **`{res.get('filename')}` successfully processed and indexed!**"
                        )

                        # Result summary metrics
                        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                        res_col1.metric("Pages Loaded", res.get("pages_loaded", 0))
                        res_col2.metric("Chunks Created", res.get("chunks_created", 0))
                        res_col3.metric("Chunks Indexed", res.get("chunks_indexed", 0))
                        res_col4.metric("Duplicates Skipped", res.get("duplicates_skipped", 0))

                        # Quick action to switch filter to this newly uploaded document
                        st.session_state.selected_document_filter = res.get("filename")
                        st.info(f"👉 Target Document Filter has been automatically set to **`{res.get('filename')}`**. Head over to the **'💬 Ask & Retrieve'** tab to query your uploaded document!")

                    except requests.exceptions.RequestException as e:
                        st.error(f"Upload failed: {e}")

    st.markdown("---")
    with st.expander("✍️ Or Ingest Raw Text / Custom Snippet Directly", expanded=False):
        with st.form("manual_ingest_form"):
            doc_name = st.text_input("Document Name", placeholder="e.g., Project_Notes.txt")
            heading = st.text_input("Section Heading (Optional)", placeholder="e.g., Section 2: Methodology")
            manual_strategy = st.selectbox("Strategy Tag", options=["recursive_character", "semantic", "custom"], index=0)
            content_text = st.text_area("Passage Text", placeholder="Paste text here...", height=140)
            submit_manual = st.form_submit_button("⚡ Ingest Snippet", use_container_width=True)

            if submit_manual:
                if not doc_name.strip() or not content_text.strip():
                    st.error("Please provide both a Document Name and content text.")
                else:
                    try:
                        res = ingest_chunk_api(
                            base_url=backend_url,
                            text=content_text.strip(),
                            source_document=doc_name.strip(),
                            section_heading=heading.strip(),
                            chunking_strategy=manual_strategy,
                        )
                        st.success(f"✅ Indexed snippet! Chunks submitted: {res.get('chunks_submitted')}")
                    except Exception as e:
                        st.error(f"Ingest failed: {e}")


# =========================================================================
# Tab 3: Indexed Knowledge Base Explorer
# =========================================================================
with tab_docs:
    st.subheader("📚 Documents & Chunk Statistics")
    st.caption("Live breakdown of all documents and chunks currently indexed across ChromaDB and BM25.")

    if not health_data:
        st.warning("Connect to the backend to view indexed documents.")
    else:
        documents = get_indexed_documents(backend_url)

        if not documents:
            st.info("No documents currently indexed in ChromaDB.")
        else:
            df = pd.DataFrame(documents)
            
            # Key Metrics
            total_chunks = df["chunk_count"].sum() if "chunk_count" in df.columns else 0
            unique_docs = df["source_document"].nunique() if "source_document" in df.columns else 0

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Indexed Documents", unique_docs)
            m_col2.metric("Total Indexed Chunks", total_chunks)
            m_col3.metric("Distinct Strategies", len(available_strategies))

            st.markdown("#### Document Breakdown")
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "source_document": st.column_config.TextColumn("Source File", width="large"),
                    "chunking_strategy": st.column_config.TextColumn("Strategy", width="medium"),
                    "chunk_count": st.column_config.ProgressColumn(
                        "Chunks",
                        format="%d",
                        min_value=0,
                        max_value=int(df["chunk_count"].max() if not df.empty else 100),
                    ),
                },
                hide_index=True,
            )

            # Distribution chart
            if "source_document" in df.columns and "chunk_count" in df.columns:
                st.markdown("#### Chunks Per Document")
                chart_data = df.set_index("source_document")[["chunk_count"]]
                st.bar_chart(chart_data)
