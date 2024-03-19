#!/usr/bin/env bash

examples_dir="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$examples_dir"
export PYTHONPATH="$examples_dir"

python apan/train.py --seed 0 --prefix exp \
    --epochs 10 --bsize 600 --n-threads 64 \
    --n-nbrs 10 --n-mail 10 \
    --sampling recent "$@"
