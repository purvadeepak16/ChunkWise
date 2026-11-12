"""Reproducible small-subset BEIR benchmark runner."""
from pathlib import Path
import json
import random
import time

import pandas as pd

from .chunking import make_chunks
from .data import load_beir
from .embeddings import encode
from .evaluation import evaluate_query, summarize
from .retrieval import rank_documents
from .contextual import contextual_text, llm_contextual_texts, rank_contextual_documents


def make_subset(corpus, queries, qrels, query_limit, document_limit, seed):
    eligible = [qid for qid, rels in qrels.items() if any(v > 0 for v in rels.values())]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen_queries, positives = [], set()
    for qid in eligible:
        query_positives = {docid for docid, value in qrels[qid].items() if value > 0}
        if len(positives | query_positives) <= document_limit:
            chosen_queries.append(qid)
            positives.update(query_positives)
        if len(chosen_queries) == min(query_limit, len(eligible)):
            break
    if not chosen_queries:
        raise ValueError("No query subset fits the selected document budget.")
    distractors = [docid for docid in corpus if docid not in positives]
    rng.shuffle(distractors)
    selected_docs = sorted(positives | set(distractors[:document_limit - len(positives)]))
    selected_qrels = {qid: {doc: value for doc, value in qrels[qid].items() if doc in selected_docs}
                      for qid in chosen_queries}
    return selected_docs, chosen_queries, selected_qrels


def run_benchmark(dataset_dir, model, strategies, sizes, query_limit, document_limit, seeds, output_dir,
                  retrieval_modes=("plain",), context_model_name="google/flan-t5-small"):
    corpus, queries, qrels = load_beir(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics, costs, manifests, failures = [], [], [], []
    for seed in seeds:
        doc_ids, query_ids, subset_qrels = make_subset(corpus, queries, qrels, query_limit, document_limit, seed)
        manifest = {"seed": seed, "documents": doc_ids, "queries": query_ids, "qrels": subset_qrels}
        manifests.append(manifest)
        (output_dir / f"manifest_{Path(dataset_dir).name}_{seed}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        for strategy in strategies:
            for size in sizes:
                started = time.perf_counter()
                callback = lambda texts: encode(model, texts)
                chunks = [chunk for doc_id in doc_ids
                          for chunk in make_chunks(corpus[doc_id]["title"] + "\n" + corpus[doc_id]["text"],
                                                   doc_id, model.tokenizer, strategy, size, callback)]
                chunk_vectors = encode(model, [chunk.text for chunk in chunks])
                plain_contextual_chunks = [
                    contextual_text(corpus[chunk.document_id]["title"], chunk.text) for chunk in chunks
                ]
                contextual_chunks = plain_contextual_chunks
                if "contextual_llm" in retrieval_modes:
                    contextual_chunks = llm_contextual_texts(
                        [corpus[chunk.document_id]["title"] for chunk in chunks],
                        [chunk.text for chunk in chunks], context_model_name)
                contextual_vectors = encode(model, contextual_chunks) if any(
                    mode.startswith("contextual") for mode in retrieval_modes
                ) else None
                indexing_seconds = time.perf_counter() - started
                rows = []
                query_started = time.perf_counter()
                query_vectors = {
                    qid: vector for qid, vector in zip(
                        query_ids, encode(model, [queries[qid] for qid in query_ids])
                    )
                }
                for retrieval_mode in retrieval_modes:
                    for qid in query_ids:
                        qv = query_vectors[qid]
                        if retrieval_mode == "plain":
                            retrieved = rank_documents(chunks, chunk_vectors, qv, 20)
                        elif retrieval_mode == "contextual_embedding":
                            retrieved = rank_contextual_documents(
                                chunks, contextual_vectors, qv, queries[qid], contextual_chunks, 20, 1.0)
                        elif retrieval_mode == "contextual_bm25":
                            retrieved = rank_contextual_documents(
                                chunks, contextual_vectors, qv, queries[qid], contextual_chunks, 20, 0.0)
                        elif retrieval_mode == "contextual_rerank":
                            retrieved = rank_contextual_documents(
                                chunks, contextual_vectors, qv, queries[qid], contextual_chunks, 20, 0.7)
                        elif retrieval_mode == "contextual_llm":
                            retrieved = rank_contextual_documents(
                                chunks, contextual_vectors, qv, queries[qid], contextual_chunks, 20, 1.0)
                        else:
                            raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")
                        ranking = [x["document_id"] for x in retrieved]
                        rows.append({"dataset": Path(dataset_dir).name, "seed": seed, "strategy": strategy,
                                     "size": size, "retrieval_mode": retrieval_mode, "query_id": qid,
                                     **evaluate_query(ranking, subset_qrels[qid])})
                        if len(failures) < 20 and not set(ranking[:10]) & {
                            doc for doc, value in subset_qrels[qid].items() if value > 0
                        }:
                            failures.append({
                                "dataset": Path(dataset_dir).name, "seed": seed, "query_id": qid,
                                "query": queries[qid],
                                "gold_relevant_documents": [doc for doc, value in subset_qrels[qid].items() if value > 0],
                                "strategy": strategy, "size": size, "retrieval_mode": retrieval_mode,
                                "retrieved_top_10": ranking[:10],
                                "best_matching_chunks": [
                                    {"document_id": item["document_id"], "score": item["score"],
                                     "chunk_id": item["chunk"].chunk_id, "text": item["chunk"].text}
                                    for item in retrieved[:10]
                                ],
                            })
                metrics = pd.DataFrame(rows)
                all_metrics.extend(rows)
                costs.append({"dataset": Path(dataset_dir).name, "seed": seed, "strategy": strategy,
                              "size": size, "retrieval_modes": ",".join(retrieval_modes),
                              "documents": len(doc_ids), "queries": len(query_ids),
                              "chunks": len(chunks), "indexing_seconds": indexing_seconds,
                              "query_seconds": time.perf_counter() - query_started})
    metrics_frame = pd.DataFrame(all_metrics)
    metrics_frame.to_csv(output_dir / f"{Path(dataset_dir).name}_metrics.csv", index=False)
    pd.DataFrame(costs).to_csv(output_dir / f"{Path(dataset_dir).name}_costs.csv", index=False)
    (output_dir / f"{Path(dataset_dir).name}_failure_analysis.json").write_text(
        json.dumps(failures, indent=2, default=str), encoding="utf-8")
    return metrics_frame, pd.DataFrame(costs), manifests
