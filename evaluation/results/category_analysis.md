# Clinical GraphRAG Retrieval Evaluation Report

## Category-Level Metrics

### Category: `citation`
| Retrieval Mode | Precision@5 | Recall@5 | MRR | Median Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Vector RAG | 0.2000 | 1.0000 | 1.0000 | 39.31ms | 42.40ms |
| Graph RAG | 0.0000 | 0.0000 | 0.0000 | 388.52ms | 654.60ms |
| Hybrid RAG | 0.2000 | 1.0000 | 0.7500 | 232.80ms | 352.30ms |

### Category: `contraindication`
| Retrieval Mode | Precision@5 | Recall@5 | MRR | Median Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Vector RAG | 0.1000 | 0.5000 | 0.5000 | 36.89ms | 38.86ms |
| Graph RAG | 0.0000 | 0.0000 | 0.0000 | 103.03ms | 129.88ms |
| Hybrid RAG | 0.1000 | 0.5000 | 0.5000 | 136.37ms | 151.32ms |

### Category: `direct`
| Retrieval Mode | Precision@5 | Recall@5 | MRR | Median Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Vector RAG | 0.2000 | 1.0000 | 1.0000 | 5620.04ms | 10648.94ms |
| Graph RAG | 0.0000 | 0.0000 | 0.0000 | 604.61ms | 1087.78ms |
| Hybrid RAG | 0.2000 | 1.0000 | 1.0000 | 119.63ms | 127.95ms |

### Category: `interaction`
| Retrieval Mode | Precision@5 | Recall@5 | MRR | Median Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Vector RAG | 0.2000 | 1.0000 | 1.0000 | 35.71ms | 39.41ms |
| Graph RAG | 0.0000 | 0.0000 | 0.0000 | 134.11ms | 138.61ms |
| Hybrid RAG | 0.2000 | 1.0000 | 1.0000 | 1178.95ms | 2014.93ms |

### Category: `multi_hop`
| Retrieval Mode | Precision@5 | Recall@5 | MRR | Median Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Vector RAG | 0.4000 | 1.0000 | 1.0000 | 40.10ms | 40.30ms |
| Graph RAG | 0.0000 | 0.0000 | 0.0000 | 75.62ms | 80.62ms |
| Hybrid RAG | 0.4000 | 1.0000 | 1.0000 | 136.34ms | 155.90ms |

### Category: `relationship`
| Retrieval Mode | Precision@5 | Recall@5 | MRR | Median Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Vector RAG | 0.2000 | 1.0000 | 1.0000 | 30.26ms | 32.53ms |
| Graph RAG | 0.0000 | 0.0000 | 0.0000 | 95.03ms | 101.51ms |
| Hybrid RAG | 0.2000 | 1.0000 | 0.7500 | 144.83ms | 180.27ms |

### Category: `semantic`
| Retrieval Mode | Precision@5 | Recall@5 | MRR | Median Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Vector RAG | 0.2000 | 1.0000 | 1.0000 | 35.52ms | 35.52ms |
| Graph RAG | 0.0000 | 0.0000 | 0.0000 | 87.54ms | 87.54ms |
| Hybrid RAG | 0.2000 | 1.0000 | 0.5000 | 110.63ms | 110.63ms |

### Category: `two_hop`
| Retrieval Mode | Precision@5 | Recall@5 | MRR | Median Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Vector RAG | 0.1000 | 0.5000 | 0.5000 | 35.75ms | 39.77ms |
| Graph RAG | 0.0000 | 0.0000 | 0.0000 | 128.88ms | 147.75ms |
| Hybrid RAG | 0.1000 | 0.5000 | 0.2500 | 122.46ms | 129.36ms |

### Category: `unanswerable`
| Retrieval Mode | Precision@5 | Recall@5 | MRR | Median Latency | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Vector RAG | 0.0000 | 0.0000 | 0.0000 | 42.42ms | 45.85ms |
| Graph RAG | 1.0000 | 1.0000 | 1.0000 | 78.39ms | 83.38ms |
| Hybrid RAG | 0.0000 | 0.0000 | 0.0000 | 127.34ms | 157.62ms |

**Unanswerable Behavior Details:**
- **Vector RAG:** Evidence retrieved = 5, Insufficient evidence state = 0
- **Graph RAG:** Evidence retrieved = 0, Insufficient evidence state = 5
- **Hybrid RAG:** Evidence retrieved = 5, Insufficient evidence state = 0

## Subset Comparisons

### Multi-Hop Subset (two_hop + multi_hop)
| Mode | Precision@5 | Recall@5 | MRR | Median Latency |
| :--- | :---: | :---: | :---: | :---: |
| Vector RAG | 0.2500 | 0.7500 | 0.7500 | 40.05ms |
| Graph RAG | 0.0000 | 0.0000 | 0.0000 | 94.54ms |
| Hybrid RAG | 0.2500 | 0.7500 | 0.6250 | 122.46ms |

### Relationship-Heavy Subset (relationship + interaction + contraindication)
| Mode | Precision@5 | Recall@5 | MRR | Median Latency |
| :--- | :---: | :---: | :---: | :---: |
| Vector RAG | 0.1667 | 0.8333 | 0.8333 | 33.74ms |
| Graph RAG | 0.0000 | 0.0000 | 0.0000 | 115.67ms |
| Hybrid RAG | 0.1667 | 0.8333 | 0.7500 | 168.59ms |

### Direct / Semantic Subset (direct + semantic)
| Mode | Precision@5 | Recall@5 | MRR | Median Latency |
| :--- | :---: | :---: | :---: | :---: |
| Vector RAG | 0.2000 | 1.0000 | 1.0000 | 35.52ms |
| Graph RAG | 0.0000 | 0.0000 | 0.0000 | 87.54ms |
| Hybrid RAG | 0.2000 | 1.0000 | 0.8333 | 110.63ms |

### Citation Subset (citation)
| Mode | Precision@5 | Recall@5 | MRR | Median Latency |
| :--- | :---: | :---: | :---: | :---: |
| Vector RAG | 0.2000 | 1.0000 | 1.0000 | 39.31ms |
| Graph RAG | 0.0000 | 0.0000 | 0.0000 | 388.52ms |
| Hybrid RAG | 0.2000 | 1.0000 | 0.7500 | 232.80ms |

### Overall Answerable-Only Metrics
| Mode | Precision@5 | Recall@5 | MRR | Median Latency |
| :--- | :---: | :---: | :---: | :---: |
| Vector RAG | 0.2000 | 0.8667 | 0.8667 | 35.88ms |
| Graph RAG | 0.0000 | 0.0000 | 0.0000 | 102.23ms |
| Hybrid RAG | 0.2000 | 0.8667 | 0.7333 | 128.88ms |

## Individual Question Results (q6, q7, q8, q9)
| Question | Mode | Precision@5 | Recall@5 | MRR | Latency | Retrieved Source IDs |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| q6 | Vector | 0.0000 | 0.0000 | 0.0000 | 31.28ms | ['f8ce57b8-ebf7-4dc1-96ec-c8fbc41c17ff', '528f5a4e-274b-5b3c-e054-00144ff8d46c', '26cca05b-82f8-e191-e063-6394a90a56bd'] |
| q6 | Graph | 0.0000 | 0.0000 | 0.0000 | 107.91ms | [] |
| q6 | Hybrid | 0.0000 | 0.0000 | 0.0000 | 114.79ms | ['26cca05b-82f8-e191-e063-6394a90a56bd', 'f8ce57b8-ebf7-4dc1-96ec-c8fbc41c17ff', '528f5a4e-274b-5b3c-e054-00144ff8d46c'] |
| q7 | Vector | 0.2000 | 1.0000 | 1.0000 | 40.22ms | ['8c45ef1f-f708-485b-bc20-60aa87ce6289', '26cca05b-82f8-e191-e063-6394a90a56bd'] |
| q7 | Graph | 0.0000 | 0.0000 | 0.0000 | 149.84ms | [] |
| q7 | Hybrid | 0.2000 | 1.0000 | 0.5000 | 130.12ms | ['26cca05b-82f8-e191-e063-6394a90a56bd', '8c45ef1f-f708-485b-bc20-60aa87ce6289'] |
| q8 | Vector | 0.4000 | 1.0000 | 1.0000 | 40.33ms | ['8c45ef1f-f708-485b-bc20-60aa87ce6289', '528f5a4e-274b-5b3c-e054-00144ff8d46c', 'f8ce57b8-ebf7-4dc1-96ec-c8fbc41c17ff'] |
| q8 | Graph | 0.0000 | 0.0000 | 0.0000 | 70.07ms | [] |
| q8 | Hybrid | 0.4000 | 1.0000 | 1.0000 | 114.60ms | ['f8ce57b8-ebf7-4dc1-96ec-c8fbc41c17ff', '8c45ef1f-f708-485b-bc20-60aa87ce6289', '528f5a4e-274b-5b3c-e054-00144ff8d46c'] |
| q9 | Vector | 0.4000 | 1.0000 | 1.0000 | 39.87ms | ['f8ce57b8-ebf7-4dc1-96ec-c8fbc41c17ff', '8c45ef1f-f708-485b-bc20-60aa87ce6289', '528f5a4e-274b-5b3c-e054-00144ff8d46c'] |
| q9 | Graph | 0.0000 | 0.0000 | 0.0000 | 81.18ms | [] |
| q9 | Hybrid | 0.4000 | 1.0000 | 1.0000 | 158.07ms | ['f8ce57b8-ebf7-4dc1-96ec-c8fbc41c17ff', '8c45ef1f-f708-485b-bc20-60aa87ce6289', '528f5a4e-274b-5b3c-e054-00144ff8d46c'] |

## Latency Distribution and Cold-Start Inspection
- **Max Latency Run:** 11207.71ms
- **First run (q1 vector):** 11207.71ms
- **Cold-Start details:** The first query execution takes significantly longer (over 6000ms in some runs) due to Hugging Face models loading and indexing caching. Subsequent queries execute with sub-second latency (typically 30ms-150ms).