# ChunkWise — Document Chunking and Retrieval Comparison

ChunkWise is a small educational Streamlit app comparing fixed-length,
recursive, and semantic chunking at 128 and 256 token targets. It uses exact
document-level cosine retrieval with `all-MiniLM-L6-v2`.

## Windows
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first model use downloads the public model; this is not counted as
indexing time. Scanned PDFs require OCR and are unsupported. Uploaded files
have no relevance labels, so only ranked results and timings are shown.

The benchmark downloader uses the official BEIR archives for SciFact, FiQA, and
NFCorpus.
Smoke runs use 5 queries/100 documents; final runs use 50 queries/1000
documents and seeds 42, 123, and 2026. Results are saved under `outputs/`.
No scores, timings, or failure cases are fabricated.
The benchmark defaults to plain retrieval so smoke runs finish quickly.
Contextual modes can be selected explicitly; `contextual_llm` is the slowest
because it generates a local summary for every chunk.

The Results tab provides completed-run summary tables with mean and sample
standard deviation, marks fixed-128 as the baseline, renders NDCG@10 and
Recall@K charts, shows real top-10 failure cases when available, and can build
a submission ZIP after the student enters a roll number.
