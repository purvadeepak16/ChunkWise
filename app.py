"""ChunkWise Streamlit application."""
from pathlib import Path
import io
import hashlib
import json
import time
import uuid

import pandas as pd
import streamlit as st
import yaml
import plotly.express as px

from src.chunking import make_chunks
from src.data import download_dataset
from src.embeddings import encode, load_model
from src.retrieval import rank_documents
from src.reporting import generate_report
from src.evaluation import evaluate_query
from src.benchmark import run_benchmark
from src.exports import make_submission_zip

ROOT = Path(__file__).parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

st.set_page_config(page_title="ChunkWise", page_icon="📚", layout="wide",
                   initial_sidebar_state="expanded")


def apply_theme():
    st.markdown("""
    <style>
    :root { --ink: #111111; --paper: #fffdf5; --yellow: #f9d949;
            --pink: #ff6b9d; --cyan: #62e6e2; --lime: #b7ef45; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stHeader"] { background: var(--paper); }
    [data-testid="stSidebar"] { background: var(--cyan); border-right: 4px solid var(--ink); }
    [data-testid="stSidebar"] * { color: var(--ink); }
    h1, h2, h3 { color: var(--ink); font-weight: 900; letter-spacing: -0.045em; }
    .hero { padding: 1.45rem 1.6rem; color: var(--ink); background: var(--yellow);
            border: 4px solid var(--ink); box-shadow: 9px 9px 0 var(--ink);
            margin: 0 .45rem 1.7rem 0; }
    .hero h1 { color: var(--ink); margin: 0; font-size: 2.5rem; font-weight: 950; }
    .hero p { margin: .35rem 0 0; font-weight: 700; font-size: 1.02rem; }
    [data-testid="stMetric"] { background: white; border: 3px solid var(--ink);
                                border-radius: 0; padding: 12px;
                                box-shadow: 5px 5px 0 var(--ink); }
    [data-testid="stMetricValue"] { font-weight: 900; }
    .stButton > button { border: 3px solid var(--ink); border-radius: 0;
                         background: var(--pink); color: var(--ink);
                         box-shadow: 4px 4px 0 var(--ink); font-weight: 900; }
    .stButton > button:hover { background: var(--lime); color: var(--ink);
                               border-color: var(--ink); transform: translate(2px,2px);
                               box-shadow: 2px 2px 0 var(--ink); }
    .stTextInput input, .stNumberInput input, [data-baseweb="select"] > div,
    [data-testid="stFileUploaderDropzone"] { border: 3px solid var(--ink);
                                               border-radius: 0; background: white; }
    [data-testid="stFileUploaderDropzone"] { box-shadow: 5px 5px 0 var(--ink); }
    [data-testid="stExpander"] { border: 3px solid var(--ink); border-radius: 0;
                                 background: white; box-shadow: 4px 4px 0 var(--ink); }
    [data-testid="stDataFrame"] { border: 3px solid var(--ink); box-shadow: 5px 5px 0 var(--ink); }
    [data-testid="stTabs"] button { font-weight: 900; color: var(--ink); }
    [data-testid="stTabs"] button[aria-selected="true"] { background: var(--lime);
                                                          border: 3px solid var(--ink); }
    .stCaption { color: #333; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def model():
    return load_model(CONFIG["model_name"])


def extract(upload):
    if upload.type == "text/plain":
        return upload.getvalue().decode("utf-8", errors="replace")
    from pypdf import PdfReader
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(upload.getvalue())).pages)


def document_key(upload):
    digest = hashlib.sha1(upload.getvalue()).hexdigest()[:8]
    return f"{digest}-{Path(upload.name).name}"


def build_index(documents, strategies, sizes):
    encoder = model()
    tokenizer = encoder.tokenizer
    index, timings = {}, {}
    for strategy in strategies:
        for size in sizes:
            started = time.perf_counter()
            callback = lambda texts: encode(encoder, texts)
            chunks = [c for doc_id, text in documents.items()
                      for c in make_chunks(text, doc_id, tokenizer, strategy, size, callback)]
            chunk_vectors = encode(encoder, [c.text for c in chunks])
            index[(strategy, size)] = (chunks, chunk_vectors)
            timings[(strategy, size)] = {"chunking_seconds": time.perf_counter() - started,
                                         "chunks": len(chunks)}
    return index, timings


def playground():
    st.subheader("Upload, index, and search your documents")
    st.caption("Local, private, exact retrieval — no API key or database required.")
    uploads = st.file_uploader("Upload TXT or text-based PDF files", ["txt", "pdf"], accept_multiple_files=True)
    documents = {}
    previews = {}
    for upload in uploads or []:
        if upload.size > CONFIG["max_upload_mb"] * 1024 * 1024:
            st.error(f"{upload.name} exceeds the {CONFIG['max_upload_mb']} MB limit.")
            continue
        try:
            text = extract(upload)
            if text.strip():
                key = document_key(upload)
                documents[key] = text
                previews[key] = upload.name
                with st.expander(f"Preview: {upload.name}"):
                    st.text(text[:3000])
            else:
                st.warning(f"{upload.name} contains no extractable text. Scanned PDFs need OCR.")
        except Exception as exc:
            st.error(f"Could not extract {upload.name}: {exc}")
    strategies = st.multiselect("Strategies", ["fixed", "recursive", "semantic"], ["fixed"])
    sizes = st.multiselect("Target token sizes", [128, 256], [128])
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents ready", len(documents))
    c2.metric("Selected strategies", len(strategies))
    c3.metric("Selected sizes", len(sizes))
    if st.button("Create Index", type="primary", disabled=not documents or not strategies or not sizes):
        with st.spinner("Indexing documents..."):
            st.session_state["index"], st.session_state["timings"] = build_index(documents, strategies, sizes)
            st.session_state["documents"] = documents
            st.session_state["document_names"] = previews
            st.session_state.pop("search_results", None)
        st.success(f"Indexed {len(documents)} document(s).")
    if "index" not in st.session_state:
        st.info("Upload documents and create an index to search.")
        return
    with st.expander("Document previews", expanded=False):
        for doc_id, text in documents.items():
            st.markdown(f"**{previews.get(doc_id, doc_id)}**")
            st.text(text[:1800])
    query = st.text_input("Natural-language query", placeholder="Example: How does semantic chunking create boundaries?")
    top_k = st.slider("Top K documents", 1, 20, CONFIG["default_top_k"])
    if st.button("Search", disabled=not query):
        qv = encode(model(), [query])[0]
        search_results = {}
        for key, (chunks, vectors) in st.session_state["index"].items():
            search_results[key] = rank_documents(chunks, vectors, qv, int(top_k))
        st.session_state["search_results"] = search_results
    if "search_results" in st.session_state:
        search_results = st.session_state["search_results"]
        st.subheader("Strategy comparison")
        overview = []
        for key, results in search_results.items():
            overview.append({"Strategy": key[0].title(), "Size": key[1],
                             "Documents returned": len(results),
                             "Best score": round(results[0]["score"], 4) if results else 0,
                             "Chunks": st.session_state["timings"][key]["chunks"],
                             "Index time (s)": round(st.session_state["timings"][key]["chunking_seconds"], 3)})
        st.dataframe(pd.DataFrame(overview), use_container_width=True, hide_index=True)
        comparison_tabs = st.tabs([f"{key[0].title()} · {key[1]} tokens" for key in search_results])
        for panel, (key, results) in zip(comparison_tabs, search_results.items()):
            with panel:
                if not results:
                    st.warning("No matching documents found.")
                    continue
                for row in results:
                    label = st.session_state.get("document_names", {}).get(row["document_id"], row["document_id"])
                    with st.container(border=True):
                        st.markdown(f"**{label}**  \nCosine similarity: `{row['score']:.4f}`")
                        st.write(row["chunk"].text)
                        st.caption(f"Chunk {row['chunk'].chunk_id} · {row['chunk'].token_count} tokens")
        with st.expander("Chunk viewer"):
            selected = st.selectbox("Choose an indexed configuration", list(st.session_state["index"]))
            chunks = st.session_state["index"][selected][0]
            st.dataframe(pd.DataFrame([{"chunk_id": c.chunk_id, "document_id": c.document_id,
                                        "tokens": c.token_count, "text": c.text} for c in chunks]),
                         use_container_width=True, hide_index=True)
    if "timings" in st.session_state:
        st.dataframe(pd.DataFrame([{"strategy": k[0], "size": k[1], **v}
                                   for k, v in st.session_state["timings"].items()]), hide_index=True)
        st.caption("Cosine similarity is a ranking score, not a confidence probability. Uploads have no relevance labels.")


def benchmark():
    st.subheader("Reproducible BEIR benchmark")
    st.caption("Use smoke mode first. Contextual LLM retrieval is optional and substantially slower than plain retrieval.")
    dataset = st.selectbox("Dataset", ["SciFact", "FiQA", "NFCorpus"])
    retrieval_modes = st.multiselect(
        "Retrieval modes",
        ["plain", "contextual_embedding", "contextual_bm25", "contextual_rerank", "contextual_llm"],
        ["plain"])
    mode = st.radio("Run type", ["Smoke test (5 queries, 100 documents)", "Final (50 queries, 1000 documents)"])
    local = ROOT / "data" / dataset.lower()
    if st.button("Download Dataset"):
        try:
            download_dataset(dataset, ROOT / "data")
            st.success(f"{dataset} downloaded to {local}.")
        except Exception as exc:
            st.error(f"Download failed: {exc}. Place an official BEIR directory at {local}.")
    if st.button("Run Benchmark"):
        if not (local / "corpus.jsonl").exists():
            st.error("Dataset is not available. Download it or provide a local fallback directory.")
        else:
            smoke = mode.startswith("Smoke")
            with st.status("Running benchmark...", expanded=True) as status:
                status.write("Selecting positive queries and seeded distractors")
                try:
                    benchmark_config = CONFIG["benchmark"]
                    metrics, costs, _ = run_benchmark(
                        local, model(), ["fixed", "recursive", "semantic"], [128, 256],
                        benchmark_config["smoke_queries"] if smoke else benchmark_config["final_queries"],
                        benchmark_config["smoke_documents"] if smoke else benchmark_config["final_documents"],
                        [42] if smoke else benchmark_config["seeds"], ROOT / "outputs", retrieval_modes,
                        CONFIG["context_model_name"])
                    status.update(label="Benchmark completed", state="complete")
                    st.success(f"Saved {len(metrics)} per-query rows and cost measurements.")
                    st.dataframe(metrics.head(20), hide_index=True)
                except Exception as exc:
                    status.update(label="Benchmark failed", state="error")
                    st.error(str(exc))


def results():
    st.subheader("Completed benchmark results")
    files = list((ROOT / "outputs").glob("*.csv"))
    if not files:
        st.info("No completed benchmark results yet. Run a benchmark first.")
        return
    metric_files = [path for path in files if path.name.endswith("_metrics.csv")]
    for path in files:
        st.subheader(path.name)
        frame = pd.read_csv(path)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", frame.to_csv(index=False), path.name, "text/csv",
                           key=f"download-{path.name}")
    if metric_files:
        metrics = pd.read_csv(metric_files[0])
        st.subheader("Summary metrics")
        summary = (metrics.groupby(["strategy", "size"])[["recall@5", "recall@10", "mrr@10", "ndcg@10"]]
                   .agg(["mean", "std"]).reset_index())
        summary.columns = [" ".join(str(x) for x in col).strip() for col in summary.columns]
        summary["baseline"] = ((summary["strategy"] == "fixed") & (summary["size"] == 128)).map(
            {True: "BASELINE", False: ""})
        st.dataframe(summary.round(4), use_container_width=True, hide_index=True)
        chart_frame = metrics.groupby(["strategy", "size"], as_index=False)["ndcg@10"].mean()
        st.plotly_chart(px.bar(chart_frame, x="strategy", y="ndcg@10", color="size",
                               barmode="group", title="F1 — Mean NDCG@10 by strategy and size"),
                        use_container_width=True)
        recall_rows = []
        for (strategy, size), group in metrics.groupby(["strategy", "size"]):
            for k in range(1, 21):
                recall_rows.append({"configuration": f"{strategy} · {size}", "k": k,
                                    "recall": group[f"recall@{k}"].mean()})
        st.plotly_chart(px.line(pd.DataFrame(recall_rows), x="k", y="recall",
                                color="configuration", markers=True,
                                title="F2 — Mean Recall@K"),
                        use_container_width=True)
        failure_files = list((ROOT / "outputs").glob("*_failure_analysis.json"))
        if failure_files:
            st.subheader("Failure analysis")
            cases = json.loads(failure_files[0].read_text(encoding="utf-8"))
            if not cases:
                st.info("No top-10 misses were found in the completed run.")
            for number, case in enumerate(cases[:5], 1):
                with st.expander(f"Case {number}: {case['query']}"):
                    st.markdown(f"**Gold relevant documents:** {', '.join(case['gold_relevant_documents'])}")
                    st.markdown(f"**Method:** {case['strategy']} / {case['size']} tokens")
                    st.write("Retrieved top 10:", case["retrieved_top_10"])
                    st.markdown("**Student interpretation:** Add your explanation here.")
                    for hit in case["best_matching_chunks"][:3]:
                        st.caption(f"{hit['document_id']} · score {hit['score']:.4f}")
                        st.write(hit["text"])
        st.subheader("Submission package")
        if st.button("Generate Evidence-Based Report"):
            report_path = generate_report(ROOT / "outputs", ROOT / "report" / "report_template.md")
            st.success(f"Report generated: {report_path.name}")
            st.download_button("Download Report", report_path.read_bytes(), report_path.name,
                               "text/markdown", key="download-report")
        roll = st.text_input("Roll number for ZIP packaging")
        if st.button("Build Submission ZIP", disabled=not roll.strip()):
            try:
                zip_path = make_submission_zip(ROOT, ROOT / "outputs" / "ChunkWise_submission.zip", roll)
                st.download_button("Download Submission ZIP", zip_path.read_bytes(),
                                   zip_path.name, "application/zip", key="submission-zip")
            except ValueError as exc:
                st.error(str(exc))


def about():
    st.subheader("About & Methodology")
    st.markdown("""
    **Chunking** splits a document into searchable pieces. Fixed chunking uses
    token windows; recursive chunking prefers sentence boundaries; semantic
    chunking uses adjacent sentence-embedding similarity as a simple boundary
    heuristic. All chunks obey the model's effective sequence budget.

    Retrieval embeds the query and chunks with the same model, normalizes them,
    and ranks unique documents by their best chunk. Benchmark metrics use
    document-level qrels: Recall measures coverage, MRR rewards the first
    relevant rank, and NDCG discounts lower ranks (qrel values > 0 are
    relevant). Results are small subsets, not full BEIR scores.

    Uploaded documents are unlabeled, so Recall/MRR/NDCG are not calculated.
    Scanned PDFs need OCR and are unsupported in this version.
    """)


apply_theme()
with st.sidebar:
    st.markdown("## 📚 CHUNKWISE")
    st.caption("Document chunking and retrieval comparison")
    st.divider()
    st.markdown("### QUICK GUIDE")
    st.markdown("1. Upload TXT/PDF files\n2. Choose strategies\n3. Create an index\n4. Search and compare")
    st.divider()
    st.caption(f"Model: `{CONFIG['model_name'].split('/')[-1]}`")
    st.caption("CPU-friendly • local files • no API key")
st.markdown("""
<div class="hero">
<h1>📚 ChunkWise</h1>
<p>See how chunking choices change document retrieval quality and speed.</p>
</div>
""", unsafe_allow_html=True)
tabs = st.tabs(["Document Playground", "Benchmark", "Results", "About & Methodology"])
with tabs[0]:
    playground()
with tabs[1]:
    benchmark()
with tabs[2]:
    results()
with tabs[3]:
    about()
