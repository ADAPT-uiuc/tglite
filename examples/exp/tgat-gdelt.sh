#!/usr/bin/env bash

examples_dir="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$examples_dir"
export PYTHONPATH="$examples_dir"

python tgat/train.py --seed 0 --prefix exp \
    --epochs 3 --bsize 4000 --n-threads 64 \
    --n-layers 2 --n-heads 2 --n-nbrs 10 \
    --sampling recent "$@"
