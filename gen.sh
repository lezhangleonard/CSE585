#!/bin/bash

SIZES=(5 20 100 1000)
HOT_RATIOS=(0.1 0.5 0.8 0.95)

for n in "${SIZES[@]}"; do
  for hot in "${HOT_RATIOS[@]}"; do
    out_file="workloads/real/w_${n}_hot_${hot}.json"
    python workload_gen_nat.py --n $n --hot $hot --out $out_file
  done
done
