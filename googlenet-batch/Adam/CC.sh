#!/bin/bash
#SBATCH --job-name=CC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=%x.out
#SBATCH --mem=160GB
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:rtx8000:2
#SBATCH --partition=rtx8000

module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate key
module load python/intel/3.8.6
module load anaconda3/2020.07
cd /scratch/xm2100/final-proj/all-sh/Adam

python3 ../../run-googlenet.py --cuda --aug=CenterCrop --numGPUs=2 --batchSize=32 --optimizer=Adam
