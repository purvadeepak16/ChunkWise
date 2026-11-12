# ChunkWise — Benchmark Report

## 1. Title and objective
Compare fixed-length, recursive, and semantic chunking at 128 and 256 token targets using document-level retrieval metrics.

## 2. Dataset and subset size
- **fiqa**: 5 queries, 100 documents.
- **nfcorpus**: 5 queries, 100 documents.
- **scifact**: 5 queries, 100 documents.

## 3. Method and baseline
The baseline is fixed chunking at 128 tokens. Retrieval deduplicates chunks at document level using the best matching chunk.

## 4. Experimental setup
Metrics are Recall@5, Recall@10, MRR@10, and NDCG@10. Costs report indexing and query time from completed runs. Contextual modes attach document titles and combine dense embeddings with BM25 where selected.

## 5. Results
| dataset | strategy | size | retrieval_mode | recall@5 | recall@10 | mrr@10 | ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fiqa | fixed | 128 | plain | 1.0 | 1.0 | 1.0 | 0.9771 |
| fiqa | fixed | 256 | plain | 1.0 | 1.0 | 1.0 | 0.9894 |
| fiqa | recursive | 128 | plain | 1.0 | 1.0 | 1.0 | 0.9894 |
| fiqa | recursive | 256 | plain | 1.0 | 1.0 | 1.0 | 0.9894 |
| fiqa | semantic | 128 | plain | 0.9333 | 1.0 | 0.9 | 0.9279 |
| fiqa | semantic | 256 | plain | 0.9333 | 1.0 | 0.9 | 0.9279 |
| nfcorpus | fixed | 128 | contextual_embedding | 0.3309 | 0.3778 | 0.8 | 0.6283 |
| nfcorpus | fixed | 256 | contextual_embedding | 0.3309 | 0.3778 | 0.8 | 0.6331 |
| nfcorpus | recursive | 128 | contextual_embedding | 0.3309 | 0.3778 | 0.8 | 0.6299 |
| nfcorpus | recursive | 256 | contextual_embedding | 0.3309 | 0.3778 | 0.8 | 0.6339 |
| nfcorpus | semantic | 128 | contextual_embedding | 0.3309 | 0.3817 | 0.8 | 0.6482 |
| nfcorpus | semantic | 256 | contextual_embedding | 0.3309 | 0.3817 | 0.7 | 0.5593 |
| scifact | fixed | 128 | contextual_bm25 | 0.8 | 0.8 | 0.7 | 0.7262 |
| scifact | fixed | 128 | contextual_embedding | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | fixed | 128 | contextual_llm | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | fixed | 128 | contextual_rerank | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | fixed | 128 | plain | 1.0 | 1.0 | 0.9 | 0.9262 |
| scifact | fixed | 256 | contextual_bm25 | 0.8 | 1.0 | 0.725 | 0.7893 |
| scifact | fixed | 256 | contextual_embedding | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | fixed | 256 | contextual_llm | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | fixed | 256 | contextual_rerank | 1.0 | 1.0 | 0.9 | 0.9262 |
| scifact | fixed | 256 | plain | 1.0 | 1.0 | 0.8667 | 0.9 |
| scifact | recursive | 128 | contextual_bm25 | 0.8 | 0.8 | 0.8 | 0.8 |
| scifact | recursive | 128 | contextual_embedding | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | recursive | 128 | contextual_llm | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | recursive | 128 | contextual_rerank | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | recursive | 128 | plain | 1.0 | 1.0 | 0.9 | 0.9262 |
| scifact | recursive | 256 | contextual_bm25 | 0.8 | 1.0 | 0.8222 | 0.8602 |
| scifact | recursive | 256 | contextual_embedding | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | recursive | 256 | contextual_llm | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | recursive | 256 | contextual_rerank | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | recursive | 256 | plain | 1.0 | 1.0 | 0.85 | 0.8861 |
| scifact | semantic | 128 | contextual_bm25 | 0.8 | 0.8 | 0.8 | 0.8 |
| scifact | semantic | 128 | contextual_embedding | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | semantic | 128 | contextual_llm | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | semantic | 128 | contextual_rerank | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | semantic | 128 | plain | 1.0 | 1.0 | 0.9 | 0.9262 |
| scifact | semantic | 256 | contextual_bm25 | 0.8 | 0.8 | 0.7 | 0.7262 |
| scifact | semantic | 256 | contextual_embedding | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | semantic | 256 | contextual_llm | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | semantic | 256 | contextual_rerank | 1.0 | 1.0 | 1.0 | 1.0 |
| scifact | semantic | 256 | plain | 1.0 | 1.0 | 0.9 | 0.9262 |

### Quality versus token cost
The following measurements are from the same completed runs; they are not estimates.
| dataset | strategy | size | chunks | indexing_seconds | query_seconds |
| --- | --- | --- | --- | --- | --- |
| fiqa | fixed | 128 | 216.0 | 4.2948 | 0.0896 |
| fiqa | fixed | 256 | 140.0 | 4.4774 | 0.0684 |
| fiqa | recursive | 128 | 226.0 | 4.2428 | 0.0667 |
| fiqa | recursive | 256 | 140.0 | 4.5448 | 0.0653 |
| fiqa | semantic | 128 | 817.0 | 13.2962 | 0.0929 |
| fiqa | semantic | 256 | 812.0 | 13.0541 | 0.092 |
| nfcorpus | fixed | 128 | 313.0 | 29.7524 | 0.4075 |
| nfcorpus | fixed | 256 | 183.0 | 28.0645 | 0.3589 |
| nfcorpus | recursive | 128 | 347.0 | 32.5061 | 0.3748 |
| nfcorpus | recursive | 256 | 186.0 | 29.6386 | 0.3206 |
| nfcorpus | semantic | 128 | 746.0 | 71.7698 | 0.4768 |
| nfcorpus | semantic | 256 | 691.0 | 104.4833 | 0.5279 |
| scifact | fixed | 128 | 331.0 | 452.5107 | 1.3092 |
| scifact | fixed | 256 | 188.0 | 203.2761 | 1.1587 |
| scifact | recursive | 128 | 371.0 | 323.6613 | 1.5488 |
| scifact | recursive | 256 | 195.0 | 163.9663 | 1.6847 |
| scifact | semantic | 128 | 702.0 | 736.2098 | 2.3794 |
| scifact | semantic | 256 | 648.0 | 6480.8912 | 0.6184 |

## 6. Analysis and five real cases
Mean NDCG@10 ranking (higher is better):
- `recursive`, 256 tokens: 0.9100
- `recursive`, 128 tokens: 0.9065
- `semantic`, 128 tokens: 0.9003
- `fixed`, 128 tokens: 0.8940
- `fixed`, 256 tokens: 0.8911
- `semantic`, 256 tokens: 0.8771

On the observed SciFact smoke subset, semantic chunking had the highest mean NDCG@10, while fixed and recursive chunking had the same lower mean. This is not sufficient to generalize to other document types until FiQA and NFCorpus runs are completed.

## 7. Limitations
- Results are limited to datasets and subsets represented by completed output files.
- Uploaded documents have no relevance labels.
- Contextual retrieval requires rerunning the benchmark with the contextual modes enabled.

## 8. References
- BEIR benchmark datasets: https://github.com/beir-cellar/beir
