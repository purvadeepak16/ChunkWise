"""Token-bounded chunking strategies used by both workflows."""
from dataclasses import dataclass
import re
from typing import Callable, List

import numpy as np


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    token_count: int
    strategy: str
    target_size: int


def effective_budget(model_tokenizer, requested: int) -> int:
    """Reserve two special-token slots and never exceed the model limit."""
    limit = getattr(model_tokenizer, "model_max_length", requested + 2)
    if limit is None or limit > 100000:
        limit = requested + 2
    return max(1, min(requested, int(limit) - 2))


def _tokens(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False, truncation=False)


def _decode(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def fixed_chunks(text: str, document_id: str, tokenizer, target: int) -> List[Chunk]:
    budget = effective_budget(tokenizer, target)
    ids = _tokens(tokenizer, text)
    return [
        Chunk(f"{document_id}-fixed-{i}", document_id, _decode(tokenizer, ids[i:i + budget]),
              min(budget, len(ids) - i), "fixed", target)
        for i in range(0, len(ids), budget)
        if _decode(tokenizer, ids[i:i + budget])
    ]


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _bounded_units(text: str, tokenizer, budget: int) -> List[str]:
    result: List[str] = []
    for unit in _split_sentences(text):
        ids = _tokens(tokenizer, unit)
        if len(ids) <= budget:
            result.append(unit)
        else:
            result.extend(_decode(tokenizer, ids[i:i + budget]) for i in range(0, len(ids), budget))
    return [u for u in result if u]


def recursive_chunks(text: str, document_id: str, tokenizer, target: int) -> List[Chunk]:
    budget = effective_budget(tokenizer, target)
    chunks, current, count = [], [], 0
    for unit in _bounded_units(text, tokenizer, budget):
        n = len(_tokens(tokenizer, unit))
        if current and count + n > budget:
            value = " ".join(current).strip()
            chunks.append(Chunk(f"{document_id}-recursive-{len(chunks)}", document_id, value,
                                len(_tokens(tokenizer, value)), "recursive", target))
            current, count = [], 0
        current.append(unit)
        count += n
    if current:
        value = " ".join(current).strip()
        chunks.append(Chunk(f"{document_id}-recursive-{len(chunks)}", document_id, value,
                            len(_tokens(tokenizer, value)), "recursive", target))
    return chunks


def semantic_chunks(
    text: str, document_id: str, tokenizer, target: int,
    embed_sentences: Callable[[List[str]], np.ndarray], threshold: float = 0.55,
) -> List[Chunk]:
    """A transparent heuristic: cut before a sentence whose neighbor is dissimilar."""
    budget = effective_budget(tokenizer, target)
    units = _bounded_units(text, tokenizer, budget)
    if not units:
        return []
    vectors = np.asarray(embed_sentences(units), dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    boundaries = [False] + [float(np.dot(vectors[i - 1], vectors[i])) < threshold
                             for i in range(1, len(units))]
    groups, current = [], []
    for unit, boundary in zip(units, boundaries):
        if boundary and current:
            groups.append(" ".join(current))
            current = []
        current.append(unit)
    if current:
        groups.append(" ".join(current))
    output: List[Chunk] = []
    for group in groups:
        ids = _tokens(tokenizer, group)
        for i in range(0, len(ids), budget):
            part = _decode(tokenizer, ids[i:i + budget])
            if part:
                output.append(Chunk(f"{document_id}-semantic-{len(output)}", document_id, part,
                                    len(_tokens(tokenizer, part)), "semantic", target))
    return output


def make_chunks(text, document_id, tokenizer, strategy, target, embed_sentences=None, threshold=0.55):
    if not text or not text.strip():
        return []
    if strategy == "fixed":
        return fixed_chunks(text, document_id, tokenizer, target)
    if strategy == "recursive":
        return recursive_chunks(text, document_id, tokenizer, target)
    if strategy == "semantic":
        if embed_sentences is None:
            raise ValueError("Semantic chunking requires an embedding callback.")
        return semantic_chunks(text, document_id, tokenizer, target, embed_sentences, threshold)
    raise ValueError(f"Unknown strategy: {strategy}")
