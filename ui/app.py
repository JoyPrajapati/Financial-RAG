import streamlit as st
import requests
import json
import time

# ── Config ────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="Financial RAG — 10-K Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .answer-box {
        background: #f0f7ff;
        border-left: 4px solid #1f4e79;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        font-size: 0.95rem;
        line-height: 1.7;
        margin-bottom: 1rem;
    }
    .source-card {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
    }
    .source-tag {
        display: inline-block;
        background: #1f4e79;
        color: white;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        margin-right: 6px;
    }
    .metric-card {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .status-ok {
        color: #2e7d32;
        font-weight: 600;
    }
    .status-fail {
        color: #c62828;
        font-weight: 600;
    }
    .query-history-item {
        background: #f9f9f9;
        border: 1px solid #eee;
        border-radius: 6px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.4rem;
        cursor: pointer;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ─────────────────────────────────────
if "ingested_file" not in st.session_state:
    st.session_state.ingested_file = None
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_queries": 0,
        "avg_response_time": 0.0,
        "total_response_time": 0.0,
        "num_sources_history": []
    }

# ── Helper Functions ───────────────────────────────────────
def check_api_health() -> bool:
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=3)
        return r.status_code == 200
    except:
        return False

def ingest_file(file) -> dict:
    file_bytes = file.getvalue()
    files = {"file": (file.name, file_bytes, "application/octet-stream")}
    r = requests.post(f"{API_BASE}/ingest", files=files, timeout=300)
    r.raise_for_status()
    return r.json()

def query_api(question: str, use_multi_query: bool,
              use_reranker: bool, top_k_retrieve: int,
              top_k_final: int) -> dict:
    payload = {
        "question": question,
        "use_multi_query": use_multi_query,
        "use_reranker": use_reranker,
        "top_k_retrieve": top_k_retrieve,
        "top_k_final": top_k_final
    }
    r = requests.post(f"{API_BASE}/query", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def format_section_name(section: str) -> str:
    mapping = {
        "item_1":  "Item 1 — Business",
        "item_1a": "Item 1A — Risk Factors",
        "item_7":  "Item 7 — MD&A",
        "item_7a": "Item 7A — Market Risk",
        "full_text": "Full Text"
    }
    return mapping.get(section, section or "Unknown")

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Financial RAG")
    st.markdown("---")

    # API Health
    api_ok = check_api_health()
    if api_ok:
        st.markdown(
            '<p class="status-ok">🟢 API Connected</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<p class="status-fail">🔴 API Offline — start uvicorn</p>',
            unsafe_allow_html=True
        )
        st.code("uvicorn api.main:app --reload --port 8000")

    st.markdown("---")

    # Ingestion Status
    st.markdown("### 📄 Ingested Document")
    if st.session_state.ingested_file:
        st.success(f"✅ {st.session_state.ingested_file}")
        st.caption(f"Chunks indexed: {st.session_state.total_chunks}")
    else:
        st.warning("No document ingested yet")

    st.markdown("---")


    st.markdown("---")
    st.markdown("### 🗂️ Query Scope")

    # Fetch indexed file list from API
    try:
        r = requests.get(f"{API_BASE}/indexed-files", timeout=5)
        if r.status_code == 200:
            indexed_files = r.json().get("files", [])
        else:
            indexed_files = []
    except:
        indexed_files = []

    if indexed_files:
        file_options = ["🔍 All indexed files"] + indexed_files
        selected = st.selectbox("Query which document?", file_options)
        st.session_state.source_filter = (
            None if selected == "🔍 All indexed files" else selected
        )
        if st.session_state.source_filter:
            st.caption(f"Filtering: **{st.session_state.source_filter}**")
    else:
        st.session_state.source_filter = None
        st.caption("No documents indexed yet")


    # Query Settings
    st.markdown("### ⚙️ Query Settings")
    use_multi_query = st.toggle("Multi-Query Retrieval", value=False)
    use_reranker = st.toggle("Re-ranker", value=False)
    top_k_retrieve = st.slider("Retrieve top-k candidates", 5, 30, 10)
    top_k_final = st.slider("Final top-k for LLM", 1, 10, 5)

    st.markdown("---")

    # Session Metrics
    st.markdown("### 📈 Session Metrics")
    m = st.session_state.metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", m["total_queries"])
    with col2:
        avg = round(m["avg_response_time"], 1)
        st.metric("Avg Time", f"{avg}s")

    if m["num_sources_history"]:
        avg_sources = round(
            sum(m["num_sources_history"]) / len(m["num_sources_history"]), 1
        )
        st.metric("Avg Sources", avg_sources)

    st.markdown("---")

    # Query History
    if st.session_state.query_history:
        st.markdown("### 🕓 Query History")
        for i, item in enumerate(
            reversed(st.session_state.query_history[-5:])
        ):
            st.markdown(
                f'<div class="query-history-item">'
                f'<b>Q{len(st.session_state.query_history) - i}:</b> '
                f'{item["question"][:60]}...</div>',
                unsafe_allow_html=True
            )

# ── Main Area ──────────────────────────────────────────────
st.markdown(
    '<p class="main-header">📊 Financial 10-K Analyzer</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">'
    'RAG system powered by DeepSeek-R1 · ChromaDB · '
    'paraphrase-MiniLM-L12-v2'
    '</p>',
    unsafe_allow_html=True
)

# ── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📤 Upload & Ingest", "💬 Query", "📊 Analytics"])

# ════════════════════════════════════════
# TAB 1 — Upload & Ingest
# ════════════════════════════════════════
with tab1:
    st.markdown("### Upload a 10-K PDF")
    st.caption(
        "Download your 10-K PDF from SEC EDGAR "
        "(https://www.sec.gov/cgi-bin/browse-edgar) "
        "and upload it here."
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "xlsx", "txt"],
        help="Supported: PDF, Word (DOCX), Excel (XLSX), Text (TXT)"
    )

    if uploaded_file:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📄 **File:** {uploaded_file.name}")
        with col2:
            size_kb = round(len(uploaded_file.getvalue()) / 1024, 1)
            st.info(f"📦 **Size:** {size_kb} KB")
        with col3:
            st.info("**Type:** PDF")

        if st.button("🚀 Ingest Document", type="primary", use_container_width=True):
            if not api_ok:
                st.error("API is offline. Start uvicorn first.")
            else:
                with st.spinner("Extracting text → Parsing sections → Chunking → Embedding → Indexing..."):
                    try:
                        start = time.time()
                        result = ingest_file(uploaded_file)
                        elapsed = round(time.time() - start, 1)

                        st.session_state.ingested_file = result["filename"]
                        st.session_state.total_chunks = result["total_chunks"]

                        st.success(
                            f"✅ Successfully ingested **{result['filename']}** "
                            f"in {elapsed}s"
                        )

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Status", result["status"].upper())
                        with col2:
                            st.metric("Chunks Indexed", result["total_chunks"])
                        with col3:
                            st.metric("Time Taken", f"{elapsed}s")

                        st.balloons()

                    except requests.exceptions.HTTPError as e:
                        st.error(f"Ingest failed: {e.response.json().get('detail', str(e))}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    else:
        st.markdown("""
        #### How to get a 10-K PDF:
        1. Go to **https://www.sec.gov/cgi-bin/browse-edgar**
        2. Enter company name (e.g. Apple, Microsoft, Google)
        3. Select **10-K** from the filing type dropdown
        4. Click on the latest filing → Download the PDF
        5. Upload it here
        """)

# ════════════════════════════════════════
# TAB 2 — Query
# ════════════════════════════════════════
with tab2:
    st.markdown("### Ask a Question")

    if not st.session_state.ingested_file:
        st.warning(
            "⚠️ No document ingested yet. "
            "Go to the **Upload & Ingest** tab first."
        )
    else:
        st.caption(
            f"Querying: **{st.session_state.ingested_file}** "
            f"({st.session_state.total_chunks} chunks)"
        )

    # Suggested questions
    st.markdown("**💡 Try one of these:**")
    suggested = [
        "What are the main risk factors this company faces?",
        "How did revenue change compared to last year?",
        "What markets does the company compete in?",
        "What are the company's main products and services?",
        "What are the quantitative market risk disclosures?",
    ]

    cols = st.columns(2)
    selected_suggestion = None
    for i, suggestion in enumerate(suggested):
        with cols[i % 2]:
            if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                selected_suggestion = suggestion

    st.markdown("---")

    # Query input
    question = st.text_area(
        "Or type your own question:",
        value=selected_suggestion or "",
        height=80,
        placeholder="e.g. What are the main risk factors Apple faces?"
    )

    if st.button("🔍 Search & Answer", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not api_ok:
            st.error("API is offline.")
        elif not st.session_state.ingested_file:
            st.error("Please ingest a PDF first.")
        else:
            with st.spinner("Retrieving relevant chunks → Generating answer..."):
                try:
                    start = time.time()
                    result = query_api(
                        question=question,
                        use_multi_query=use_multi_query,
                        use_reranker=use_reranker,
                        top_k_retrieve=top_k_retrieve,
                        top_k_final=top_k_final
                    )
                    elapsed = round(time.time() - start, 1)

                    # Update metrics
                    m = st.session_state.metrics
                    m["total_queries"] += 1
                    m["total_response_time"] += elapsed
                    m["avg_response_time"] = (
                        m["total_response_time"] / m["total_queries"]
                    )
                    m["num_sources_history"].append(result["num_sources"])

                    # Save to history
                    st.session_state.query_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "num_sources": result["num_sources"],
                        "elapsed": elapsed,
                        "settings": {
                            "multi_query": use_multi_query,
                            "reranker": use_reranker,
                            "top_k_retrieve": top_k_retrieve,
                            "top_k_final": top_k_final
                        }
                    })

                    st.session_state.last_result = {
                        **result,
                        "elapsed": elapsed,
                        "question": question
                    }

                except requests.exceptions.HTTPError as e:
                    st.error(
                        f"Query failed: {e.response.json().get('detail', str(e))}"
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Display last result
    if st.session_state.last_result:
        r = st.session_state.last_result

        st.markdown("---")
        st.markdown("### 🤖 Answer")

        # Response metadata row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Response Time", f"{r['elapsed']}s")
        with col2:
            st.metric("Sources Used", r["num_sources"])
        with col3:
            st.metric("Model", "DeepSeek-R1")
        with col4:
            st.metric(
                "Retrieval",
                "Multi-Q" if use_multi_query else "Single-Q"
            )

        # Answer box
        st.markdown(
            f'<div class="answer-box">{r["answer"]}</div>',
            unsafe_allow_html=True
        )

        # Sources
        if r["sources"]:
            st.markdown("### 📚 Source Chunks")
            st.caption(
                "These are the exact passages from the 10-K "
                "that were used to generate the answer."
            )
            for i, source in enumerate(r["sources"], 1):
                section_label = format_section_name(source.get("section"))
                filename = source.get("source", "unknown")
                snippet = source.get("text_snippet", "")
                with st.expander(
                    f"Source {i} — {section_label} | {filename}"
                ):
                    st.markdown(
                        f'<span class="source-tag">{section_label}</span>'
                        f'<span class="source-tag">{filename}</span>',
                        unsafe_allow_html=True
                    )
                    st.markdown("**Text snippet:**")
                    st.markdown(f"> {snippet}")
        else:
            st.info(
                "No sources returned. "
                "This may mean the ChromaDB index is empty — "
                "try re-ingesting the PDF."
            )

        # Raw JSON toggle
        with st.expander("🔧 Raw API Response (for debugging)"):
            st.json(r)

# ════════════════════════════════════════
# TAB 3 — Analytics
# ════════════════════════════════════════
with tab3:
    st.markdown("### 📊 Session Analytics")

    if not st.session_state.query_history:
        st.info("No queries yet. Go to the Query tab and ask some questions.")
    else:
        m = st.session_state.metrics
        history = st.session_state.query_history

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Queries", m["total_queries"])
        with col2:
            st.metric("Avg Response Time", f"{round(m['avg_response_time'], 1)}s")
        with col3:
            avg_src = round(
                sum(m["num_sources_history"]) / len(m["num_sources_history"]), 1
            ) if m["num_sources_history"] else 0
            st.metric("Avg Sources Retrieved", avg_src)
        with col4:
            st.metric(
                "Document",
                st.session_state.ingested_file or "None"
            )

        st.markdown("---")

        # Response time chart
        st.markdown("#### ⏱️ Response Time per Query")
        times = [h["elapsed"] for h in history]
        labels = [f"Q{i+1}" for i in range(len(history))]
        chart_data = {"Query": labels, "Response Time (s)": times}
        import pandas as pd
        df_times = pd.DataFrame(chart_data).set_index("Query")
        st.bar_chart(df_times)

        # Sources per query chart
        st.markdown("#### 📚 Sources Retrieved per Query")
        sources_counts = [h["num_sources"] for h in history]
        df_sources = pd.DataFrame(
            {"Query": labels, "Sources": sources_counts}
        ).set_index("Query")
        st.bar_chart(df_sources)

        st.markdown("---")

        # Full query history table
        st.markdown("#### 📋 Full Query History")
        for i, h in enumerate(reversed(history), 1):
            with st.expander(
                f"Q{len(history) - i + 1}: {h['question'][:80]}..."
            ):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("**Question:**")
                    st.write(h["question"])
                    st.markdown("**Answer:**")
                    st.markdown(
                        f'<div class="answer-box">{h["answer"]}</div>',
                        unsafe_allow_html=True
                    )
                with col2:
                    st.markdown("**Settings:**")
                    st.write(f"⏱ {h['elapsed']}s")
                    st.write(f"📚 {h['num_sources']} sources")
                    s = h["settings"]
                    st.write(
                        f"Multi-Q: {'✅' if s['multi_query'] else '❌'}"
                    )
                    st.write(
                        f"Reranker: {'✅' if s['reranker'] else '❌'}"
                    )
                    st.write(f"Top-K: {s['top_k_retrieve']} → {s['top_k_final']}")

        # Export history
        st.markdown("---")
        st.markdown("#### 💾 Export Query History")
        export_data = json.dumps(history, indent=2)
        st.download_button(
            label="⬇️ Download as JSON",
            data=export_data,
            file_name="query_history.json",
            mime="application/json"
        )