import numpy as np
from src.chunking import Chunk
from src.retrieval import rank_documents
from src.contextual import bm25_scores, contextual_text


def test_document_level_uses_best_chunk():
    chunks = [Chunk("a-0", "a", "x", 1, "fixed", 128),
              Chunk("b-0", "b", "y", 1, "fixed", 128),
              Chunk("a-1", "a", "z", 1, "fixed", 128)]
    result = rank_documents(chunks, np.array([[1, 0], [0, 1], [0.9, 0.1]]), np.array([1, 0]), 2)
    assert [row["document_id"] for row in result] == ["a", "b"]
    assert result[0]["chunk"].chunk_id == "a-0"


def test_contextual_helpers_preserve_context_and_rank_terms():
    assert contextual_text("A title", "body").startswith("Document title: A title")
    scores = bm25_scores(["alpha beta", "gamma"], "alpha")
    assert scores[0] > scores[1]
