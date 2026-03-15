#!/bin/bash
#SBATCH --job-name=download_usound_data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --qos=cpu_qos
#SBATCH --chdir=/home/jgibson2/projects/breast_cancer_detection
#SBATCH --output=slurm_logs/breast_cancer_detection_%j.out
#SBATCH --error=slurm_logs/breast_cancer_detection_%j.err

set -euo pipefail

cd /home/jgibson2/projects/breast_cancer_detection
source venv312/bin/activate

python /home/jgibson2/projects/breast_cancer_detection/datasets/download_kaggle_data.py

echo "Data download / preprocessing completed."
