"""Document-level IR metrics with positive qrels."""
import math
import numpy as np


def recall_at_k(ranking, relevant, k):
    relevant = {str(x) for x, value in relevant.items() if value > 0} if isinstance(relevant, dict) else set(relevant)
    return len(set(ranking[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(ranking, relevant, k=10):
    relevant = {str(x) for x, value in relevant.items() if value > 0} if isinstance(relevant, dict) else set(relevant)
    for i, doc in enumerate(ranking[:k], 1):
        if str(doc) in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking, relevant, k=10):
    qrels = {str(x): float(v) for x, v in relevant.items()} if isinstance(relevant, dict) else {str(x): 1.0 for x in relevant}
    def dcg(items):
        return sum((2 ** qrels.get(str(doc), 0) - 1) / math.log2(i + 2) for i, doc in enumerate(items))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    ideal_dcg = sum((2 ** value - 1) / math.log2(i + 2) for i, value in enumerate(ideal))
    return dcg(ranking[:k]) / ideal_dcg if ideal_dcg else 0.0


def evaluate_query(ranking, qrels):
    return {
        "recall@5": recall_at_k(ranking, qrels, 5),
        "recall@10": recall_at_k(ranking, qrels, 10),
        "mrr@10": mrr_at_k(ranking, qrels),
        "ndcg@10": ndcg_at_k(ranking, qrels),
        **{f"recall@{k}": recall_at_k(ranking, qrels, k) for k in range(1, 21)},
    }


def summarize(rows, columns=None):
    columns = columns or ["recall@5", "recall@10", "mrr@10", "ndcg@10"]
    return {col: {"mean": float(np.mean([r[col] for r in rows])),
                  "std": float(np.std([r[col] for r in rows], ddof=1)) if len(rows) > 1 else 0.0}
            for col in columns}
