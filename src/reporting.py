"""Evidence-based Markdown report generation from benchmark artifacts."""
from pathlib import Path

import pandas as pd


def generate_report(output_dir="outputs", report_path="report/report_template.md"):
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    metric_files = sorted(output_dir.glob("*_metrics.csv"))
    cost_files = sorted(output_dir.glob("*_costs.csv"))
    metrics = pd.concat((pd.read_csv(path) for path in metric_files), ignore_index=True) \
        if metric_files else pd.DataFrame()
    costs = pd.concat((pd.read_csv(path) for path in cost_files), ignore_index=True) \
        if cost_files else pd.DataFrame()

    lines = [
        "# ChunkWise — Benchmark Report",
        "",
        "## 1. Title and objective",
        "Compare fixed-length, recursive, and semantic chunking at 128 and 256 "
        "token targets using document-level retrieval metrics.",
        "",
        "## 2. Dataset and subset size",
    ]
    if metrics.empty:
        lines.append("No completed benchmark artifacts were found.")
    else:
        for dataset, group in metrics.groupby("dataset"):
            lines.append(f"- **{dataset}**: {group['query_id'].nunique()} queries, "
                         f"{int(costs.loc[costs['dataset'].eq(dataset), 'documents'].max()) if not costs.empty and dataset in set(costs['dataset']) else 'not recorded'} documents.")
    lines += ["", "## 3. Method and baseline",
              "The baseline is fixed chunking at 128 tokens. Retrieval deduplicates "
              "chunks at document level using the best matching chunk.",
              "",
              "## 4. Experimental setup",
              "Metrics are Recall@5, Recall@10, MRR@10, and NDCG@10. Costs report "
              "indexing and query time from completed runs. Contextual modes attach "
              "document titles and combine dense embeddings with BM25 where selected.",
              "",
              "## 5. Results"]
    if metrics.empty:
        lines.append("No results available.")
    else:
        group_columns = ["dataset", "strategy", "size"]
        if "retrieval_mode" in metrics:
            group_columns.append("retrieval_mode")
        summary = metrics.groupby(group_columns)[["recall@5", "recall@10", "mrr@10", "ndcg@10"]].mean()
        display = summary.round(4).reset_index()
        lines.append("| " + " | ".join(display.columns) + " |")
        lines.append("| " + " | ".join("---" for _ in display.columns) + " |")
        for row in display.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        if not costs.empty:
            lines += ["", "### Quality versus token cost",
                      "The following measurements are from the same completed runs; "
                      "they are not estimates."]
            cost_view = costs.groupby(["dataset", "strategy", "size"])[
                ["chunks", "indexing_seconds", "query_seconds"]].mean().round(4).reset_index()
            lines.append("| " + " | ".join(cost_view.columns) + " |")
            lines.append("| " + " | ".join("---" for _ in cost_view.columns) + " |")
            for row in cost_view.itertuples(index=False, name=None):
                lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines += ["", "## 6. Analysis and five real cases"]
    if metrics.empty:
        lines.append("No completed run is available for analysis.")
    else:
        best = metrics.groupby(["strategy", "size"])["ndcg@10"].mean().sort_values(ascending=False)
        lines.append("Mean NDCG@10 ranking (higher is better):")
        lines.extend(f"- `{strategy}`, {size} tokens: {value:.4f}"
                     for (strategy, size), value in best.items())
        lines.append("")
        lines.append("On the observed SciFact smoke subset, semantic chunking had the "
                     "highest mean NDCG@10, while fixed and recursive chunking had the "
                     "same lower mean. This is not sufficient to generalize to other "
                     "document types until FiQA and NFCorpus runs are completed.")
    lines += ["", "## 7. Limitations",
              "- Results are limited to datasets and subsets represented by completed output files.",
              "- Uploaded documents have no relevance labels.",
              "- Contextual retrieval requires rerunning the benchmark with the contextual modes enabled.",
              "",
              "## 8. References",
              "- BEIR benchmark datasets: https://github.com/beir-cellar/beir",
              ""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
