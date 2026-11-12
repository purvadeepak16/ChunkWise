import plotly.express as px
import plotly.graph_objects as go


def ndcg_chart(frame):
    return px.bar(frame, x="strategy", y="ndcg@10", color="size", barmode="group",
                  title="F1 — NDCG@10 by strategy and size", labels={"ndcg@10": "NDCG@10"})


def recall_chart(frame):
    return px.line(frame, x="k", y="recall", color="configuration", markers=True,
                   title="F2 — Recall@K", labels={"k": "K", "recall": "Recall@K"})
