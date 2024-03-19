#!/usr/bin/env bash
#
# Script to setup environment for this repo.
#

tglite="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$tglite"

echo
echo ">> setting up environment"
echo

source ~/.conda/etc/profile.d/conda.sh
conda create -n tglite python=3.7
conda activate tglite

echo
echo ">> installing python packages"
echo

pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter==2.1.0+pt112cu116 -f https://data.pyg.org/whl/torch-1.12.1+cu116.html

echo
echo ">> installing tglite package"
echo

python setup.py develop

echo
echo ">> setting up example applications"
echo

cd "$tglite/examples"
pip install -r requirements.txt
./download-data.sh
python gen-data-files.py --data wiki-talk

echo
echo ">> done!"
