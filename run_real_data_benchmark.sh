#!/bin/bash
set -e

echo "====================================================="
echo " Running KNN Optimization Real Data Benchmark"
echo "====================================================="

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
../trackastra_env/bin/python scripts/verify_knn_real_data.py

echo "====================================================="
echo " Done! Benchmark results saved."
echo "====================================================="
