#!/bin/bash

SIZES=(50 100 200 500)
HOT_RATIOS=(0.1 0.5 0.8 0.95)

for n in "${SIZES[@]}"; do
  for hot in "${HOT_RATIOS[@]}"; do
    out_file_synthetic="workloads/synthetic/w_${n}_hot_${hot}.json"
    out_file_real="workloads/real/w_${n}_hot_${hot}.json"
    python workload_gen.py --n $n --hot $hot --out $out_file_synthetic --mode synthetic
    python workload_gen.py --n $n --hot $hot --out $out_file_real --mode real
  done
done
