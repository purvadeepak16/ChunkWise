"""Reproducible contextual retrieval helpers for the SciFact comparison."""
import math
import re
from collections import Counter

import numpy as np
from functools import lru_cache


def contextual_text(title: str, chunk_text: str) -> str:
    """Attach document-level context before embedding or lexical indexing."""
    title = title.strip()
    return f"Document title: {title}\nContent: {chunk_text}" if title else chunk_text


@lru_cache(maxsize=2)
def load_context_model(model_name: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def llm_contextual_texts(titles, chunk_texts, model_name="google/flan-t5-small", batch_size=8):
    """Generate short local-LLM context prefixes for chunks."""
    tokenizer, model = load_context_model(model_name)
    model.eval()
    outputs = []
    for start in range(0, len(chunk_texts), batch_size):
        prompts = [
            f"Summarize the topic and key claim of this scientific passage in one sentence.\n"
            f"Title: {title}\nPassage: {text}"
            for title, text in zip(titles[start:start + batch_size], chunk_texts[start:start + batch_size])
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        import torch
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=48, num_beams=2)
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return [contextual_text(title, f"{summary.strip()}\nContent: {text}")
            for title, summary, text in zip(titles, outputs, chunk_texts)]


def _terms(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_scores(documents, query, k1=1.5, b=0.75):
    """Return BM25 scores for one query over one text per document."""
    tokenized = [_terms(text) for text in documents]
    query_terms = set(_terms(query))
    lengths = np.array([len(tokens) for tokens in tokenized], dtype=float)
    average_length = float(lengths.mean()) if len(lengths) else 1.0
    document_frequency = Counter(term for tokens in tokenized for term in set(tokens))
    scores = np.zeros(len(tokenized), dtype=float)
    for index, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        for term in query_terms:
            if not counts[term]:
                continue
            idf = math.log(1 + (len(tokenized) - document_frequency[term] + 0.5) /
                           (document_frequency[term] + 0.5))
            denominator = counts[term] + k1 * (1 - b + b * lengths[index] / max(average_length, 1.0))
            scores[index] += idf * counts[term] * (k1 + 1) / denominator
    return scores


def rank_contextual_documents(chunks, chunk_vectors, query_vector, query, contextual_chunks,
                              top_k=5, dense_weight=0.7):
    """Combine contextual dense retrieval with BM25 and deduplicate by document."""
    if not chunks:
        return []
    matrix = np.asarray(chunk_vectors, dtype=float)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    query_vector = np.asarray(query_vector, dtype=float).reshape(-1)
    query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
    dense = matrix @ query_vector
    sparse = bm25_scores(contextual_chunks, query)
    if sparse.max() > sparse.min():
        sparse = (sparse - sparse.min()) / (sparse.max() - sparse.min())
    combined = dense_weight * dense + (1 - dense_weight) * sparse
    best = {}
    for chunk, score in zip(chunks, combined):
        previous = best.get(chunk.document_id)
        candidate = (float(score), chunk)
        if previous is None or candidate[0] > previous[0] or (
            candidate[0] == previous[0] and chunk.chunk_id < previous[1].chunk_id
        ):
            best[chunk.document_id] = candidate
    return [{"document_id": doc, "score": score, "chunk": chunk}
            for doc, (score, chunk) in sorted(best.items(), key=lambda item: (-item[1][0], item[0]))[:top_k]]
