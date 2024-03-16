#!/usr/bin/env bash
#
# Script to setup an AWS EC2 instance with the following expected provisioning:
#
# - AMI: Amazon Linux 2 AMI with NVIDIA TESLA GPU Driver
# - Instance Type: p3.8xlarge (32 vCPU, 244 GiB, Tesla V100 GPU)
# - Storage: at least 80 GB
#

conda_dir="$HOME/.conda"
conda_bin="$conda_dir/bin/conda"

echo
echo ">> installing conda"
echo

curl -sL -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh
bash ~/miniconda.sh -b -p "$conda_dir"
rm ~/miniconda.sh

"$conda_bin" init
"$conda_bin" config --set auto_activate_base false
source ~/.conda/etc/profile.d/conda.sh

echo
echo ">> setting up cuda11"
echo

conda create -n cuda11
conda activate cuda11
conda install -c "nvidia/label/cuda-11.8.0" cuda-libraries
echo 'export LD_LIBRARY_PATH="$HOME/.conda/envs/cuda11/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc

echo
echo ">> done! please restart your shell session"
