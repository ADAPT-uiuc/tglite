#!/usr/bin/env bash
#
# Run experiments for standard benchmarks.
#

tglite="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$tglite/examples"

echo "start: $(date)"

for data in wiki mooc reddit lastfm; do
  for model in apan jodie tgat tgn; do
    common_flags="--data $data"
    if [[ "$model" != "jodie" ]]; then
      common_flags+=" --n-threads $(nproc)"
    fi

    echo "tglite $data $model";
    "./exp/$model.sh" $common_flags
    mv out-stats.csv "out-tglite-$data-$model.csv";
    echo;
    echo "time: $(date)"
    echo;

    "./exp/$model.sh" $common_flags --move
    mv out-stats.csv "out-tglite-allgpu-$data-$model.csv";
    echo;
    echo "time: $(date)"
    echo;

    if [[ "$model" != "jodie" ]]; then
      "./exp/$model.sh" $common_flags --opt-all
      mv out-stats.csv "out-tglite-opt-$data-$model.csv";
      echo;
      echo "time: $(date)"
      echo;
    fi
  done
done

echo "end: $(date)"
