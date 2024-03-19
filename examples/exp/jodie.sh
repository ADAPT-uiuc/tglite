#!/usr/bin/env bash

examples_dir="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$examples_dir"
export PYTHONPATH="$examples_dir"

python jodie/train.py --seed 0 --prefix exp \
    --epochs 10 --bsize 600 "$@"
