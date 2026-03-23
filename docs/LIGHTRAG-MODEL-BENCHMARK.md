# LIGHTRAG Model Benchmark (RN-046)

- Generated at: `2026-03-23T18:13:48Z`
- GPU profile: `H100`
- Scope: 5 collections, 1 representative segment each, 3 models (3b/8b/32b)

## Model Summary

| Model | Runs | Success | Total Time (sec) | Avg Time (sec) | Nodes | Edges |
|---|---:|---:|---:|---:|---:|---:|
| llama3.2:3b | 5 | 5 | 490.72 | 98.14 | 204 | 94 |
| llama3.1:8b | 5 | 5 | 578.90 | 115.78 | 212 | 118 |
| qwen2.5:32b | 5 | 5 | 985.31 | 197.06 | 276 | 122 |

## Per Collection (Actual Counts)

| Collection | 3b Nodes | 3b Edges | 8b Nodes | 8b Edges | 32b Nodes | 32b Edges |
|---|---:|---:|---:|---:|---:|---:|
| ref-engineering | 22 | 21 | 25 | 16 | 26 | 17 |
| ref-finance | 57 | 3 | 41 | 24 | 67 | 20 |
| ref-negotiation | 26 | 17 | 19 | 15 | 24 | 20 |
| ref-networking | 65 | 28 | 64 | 33 | 100 | 36 |
| ref-software | 34 | 25 | 63 | 30 | 59 | 29 |

