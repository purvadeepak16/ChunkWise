"""Exact document-level cosine retrieval."""
import numpy as np


def rank_documents(chunks, chunk_vectors, query_vector, top_k=5):
    if not chunks:
        return []
    matrix = np.asarray(chunk_vectors, dtype=float)
    q = np.asarray(query_vector, dtype=float).reshape(-1)
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    scores = matrix @ q
    best = {}
    for chunk, score in zip(chunks, scores):
        previous = best.get(chunk.document_id)
        candidate = (float(score), chunk)
        if previous is None or candidate[0] > previous[0] or (
            candidate[0] == previous[0] and chunk.chunk_id < previous[1].chunk_id
        ):
            best[chunk.document_id] = candidate
    return [
        {"document_id": doc, "score": score, "chunk": chunk}
        for doc, (score, chunk) in sorted(best.items(), key=lambda x: (-x[1][0], x[0]))[:top_k]
    ]
