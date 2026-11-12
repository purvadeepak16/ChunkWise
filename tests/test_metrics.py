from src.evaluation import evaluate_query


def test_metrics_toy_case():
    result = evaluate_query(["b", "a", "c"], {"a": 1, "c": 1})
    assert result["recall@5"] == 1.0
    assert result["mrr@10"] == 0.5
    assert result["ndcg@10"] > 0
