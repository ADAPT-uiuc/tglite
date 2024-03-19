#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64     # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4
#SBATCH --time=00:10:00
#SBATCH --account=bbzw-delta-gpu
#SBATCH --job-name=jodie-wiki
#SBATCH --output=test.out
#SBATCH --error=test.err
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1
###SBATCH --gpu-bind=none     # <- or closest

source ~/.bashrc
conda deactivate
module purge
module load anaconda3_gpu
module list

conda activate tglite
conda info -e

echo "job is starting on `hostname`"

tglite="$(cd "$(dirname "$0")"; cd ..; pwd)"
cd "$tglite/examples"

echo "start: $(date)"

srun python3 \
  jodie.py \
  --seed 0 \
  --prefix exp \
  --epochs 50 \
  --bsize 2000 \
  -d wiki

echo "end $(date)"

exit
