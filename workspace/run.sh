#!/bin/bash
#SBATCH -c 4                               # Request one core
#SBATCH -t 8:00:00                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=16G                          # Memory total in MiB (for all cores)
#SBATCH -o ./logs/terminal/fairness_%j.log
#SBATCH -e ./logs/terminal/fairness_%j.log

# Activation of the env and get the cancer_id
source activate pdm

# Main process
python test_metrics.py