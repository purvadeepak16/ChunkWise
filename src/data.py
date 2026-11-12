"""Small, transparent BEIR downloader and local JSONL reader."""
from pathlib import Path
import json
import shutil
import ssl
import urllib.request
import zipfile

import certifi

URLS = {
    "SciFact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "FiQA": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "NFCorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
}


def download_dataset(name, root="data"):
    if name not in URLS:
        raise ValueError(f"Unsupported dataset: {name}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{name.lower()}.zip"
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(URLS[name], context=context) as response:
        with archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    return root / name.lower()


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_beir(root):
    root = Path(root)
    corpus = {str(x["_id"]): x for x in read_jsonl(root / "corpus.jsonl")}
    queries = {str(x["_id"]): x["text"] for x in read_jsonl(root / "queries.jsonl")}
    qrels = {}
    with open(root / "qrels" / "test.tsv", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            qid, docid, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[docid] = int(score)
    return corpus, queries, qrels
