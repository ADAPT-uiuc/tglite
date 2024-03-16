#!/usr/bin/env bash
#
# Run experiments for larger benchmarks.
#

tglite="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$tglite/examples"

echo "start: $(date)"

for data in wiki-talk gdelt; do
  for model in apan jodie tgat tgn; do
    echo "tglite $data $model";
    if [[ "$model" != "jodie" ]]; then
      "./exp/$model-gdelt.sh" --data "$data" --n-threads "$(nproc)" --opt-all
    else
      "./exp/$model-gdelt.sh" --data "$data"
    fi
    mv out-stats.csv "out-tglite-$data-$model.csv";
    echo;
    echo "time: $(date)"
    echo;
  done
done

echo "end: $(date)"
