#!/bin/bash
#SBATCH -c 4                               # Request one core
#SBATCH -t 60:00:00                        # Runtime in D-HH:MM format
#SBATCH -p medium                          # Partition to run in
#SBATCH --mem=32G                          # Memory total in MiB (for all cores)
#SBATCH -o ./logs/terminal/fairness_%j.log
#SBATCH -e ./logs/terminal/fairness_%j.log
#SBATCH --array=0-17

# Activation of the env and get the cancer_id
source activate pdm
JOB_ID=$((SLURM_ARRAY_TASK_ID))

# Main process
python pipeline.py  --allow_wandb 1 \
                    --job_id ${JOB_ID} \
                    --ID 1 \
                    \
                    --custom_subgroups 0 \
                    --add_protected_attributes 1 \
                    \
                    --model Martinez \
                    --save_model 1 \
                    \
                    --nb_epochs 500 \
                    --split_regarding_subgroups 1 \
                    --NB_STEPS 100 \
                    \
                    --save_preds 1