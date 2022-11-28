#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:40:00
#SBATCH --mem=4GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=aug4
#SBATCH --output=aug4_sgd.out

module purge
module load anaconda3/2020.07
conda activate project
cd /scratch/xh2311/hpml/HPML-Final-Proj/Alexnet

python ./augmentation4.py --optimizer sgd