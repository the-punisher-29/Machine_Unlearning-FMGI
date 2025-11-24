#!/bin/bash
#SBATCH -p phd                 # partition
#SBATCH --gres=gpu:1          # request 1 GPU
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00           # max time
#SBATCH --job-name=sisa_run
#SBATCH --output=logs/sisa_%j.out   # %j = job ID

# ----- Load Python 3.10 -----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate assi

# Create needed dirs
mkdir -p results
mkdir -p results/checkpoints
mkdir -p logs

echo "Running SISA pipeline on GPU: $(hostname)"

run_step () {
    NAME=$1
    CMD=$2

    echo "==============================="
    echo "Starting: $NAME"
    echo "Command: $CMD"
    echo "==============================="

    START=$(date +%s)

    eval $CMD

    END=$(date +%s)
    DIFF=$(( (END - START) / 60 ))

    echo "Finished: $NAME in ${DIFF} minutes"
}

# -------- Pipeline steps ---------

run_step "Baseline SISA training" \
"python -m code.cli baseline --num-shards 20 --num-slices 3 --epochs 20 --batch-size 128 --optimizer adam --device cuda --output-dir results --checkpoint-dir results/checkpoints"

run_step "Selective retraining (Micro-deletion 0.02%)" \
"python -m code.cli deletion --percent 0.0002 --num-shards 20 --num-slices 3 --epochs 20 --batch-size 128 --optimizer adam --device cuda --output-dir results --checkpoint-dir results/checkpoints"

run_step "Selective retraining after 1% deletion" \
"python -m code.cli deletion --percent 0.01 --num-shards 20 --num-slices 3 --epochs 20 --batch-size 128 --optimizer adam --device cuda --output-dir results --checkpoint-dir results/checkpoints"

run_step "Selective retraining after 5% deletion" \
"python -m code.cli deletion --percent 0.05 --num-shards 20 --num-slices 3 --epochs 20 --batch-size 128 --optimizer adam --device cuda --output-dir results --checkpoint-dir results/checkpoints"

run_step "Naive full retrain (1%)" \
"python -m code.cli naive-delete --percent 0.01 --num-shards 20 --num-slices 3 --epochs 20 --batch-size 128 --optimizer adam --device cuda --output-dir results --checkpoint-dir results/checkpoints"

run_step "Naive full retrain (5%)" \
"python -m code.cli naive-delete --percent 0.05 --num-shards 20 --num-slices 3 --epochs 20 --batch-size 128 --optimizer adam --device cuda --output-dir results --checkpoint-dir results/checkpoints"

run_step "MIA comparison (1%)" \
"python -m code.cli mia --percent 0.01 --num-shards 20 --num-slices 3 --epochs 20 --batch-size 128 --optimizer adam --device cuda --output-dir results --checkpoint-dir results/checkpoints"

run_step "MIA comparison (5%)" \
"python -m code.cli mia --percent 0.05 --num-shards 20 --num-slices 3 --epochs 20 --batch-size 128 --optimizer adam --device cuda --output-dir results --checkpoint-dir results/checkpoints"

run_step "Generate plots + Table 1" \
"python -m code.report_artifacts --results-dir results --figures-dir results/figures --table-path results/table1.csv"

echo "All SISA steps finished!"
