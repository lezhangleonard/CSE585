# Chart Map

---

## Core System Results

| ID | Chart | Description |
|----|------|-------------|
| 01 | Correctness under Contention | Shows correctness stability of DAG vs Batch as hot ratio increases |
| 02 | Throughput under Contention | Shows performance degradation or scaling under increasing contention |
| 03 | DAG Speedup vs Sequential | Core result: DAG acceleration over sequential baseline |

---

## Mechanism Analysis

| ID | Chart | Description |
|----|------|-------------|
| 04 | DAG Parallel Width | Measures extracted concurrency from DAG scheduler |
| 05 | Reasoning Latency under Contention | LLM decision cost across workloads |
| 06 | DAG Parallel Width Heatmap | Dense view of concurrency vs input size and contention |
| 13 | LLM Time Fraction | Shows how much of total execution time is spent in LLM vs system overhead (extractor + reasoner breakdown). |

---

## Tradeoffs

| ID | Chart | Description |
|----|------|-------------|
| 07 | Correctness–Throughput Tradeoff | Shows whether speed gains reduce correctness |
| 10 | No-op Ratio under Contention | Fraction of rejected or redundant updates |
| 11 | Mutation Ratio under Contention | Fraction of updates that actually modify state |

---

## Scaling Behavior

| ID | Chart | Description |
|----|------|-------------|
| 08 | Throughput Scaling with Input Size | How performance changes as input size increases |
| 12 | DAG Throughput Scaling | Scaling behavior of DAG-only executor |

---

## Distributional Analysis

| ID | Chart | Description |
|----|------|-------------|
| 09 | Latency Distribution across Workloads | Variance in LLM reasoning latency across runs |

---

# Key Insight Summary

- DAG preserves correctness under all contention levels
- DAG significantly improves throughput over sequential baseline
- Naive batching introduces correctness failures under high contention
- Parallel width adapts dynamically to workload pressure