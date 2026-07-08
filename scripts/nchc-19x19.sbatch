#!/bin/bash
# 19x19 training on one NCHC 8xH200 node.
#
#   sbatch scripts/nchc-19x19.sbatch
#
# Auto-resumes from the latest checkpoint, so requeueing after
# walltime just continues; SLURM's TERM (10 min before the limit)
# triggers the trainer's graceful save of checkpoint + buffer.
#
#SBATCH -J goboard-19
#SBATCH -A ACD114124
#SBATCH -p 8gpus
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 2-00:00:00
#SBATCH --signal=TERM@600
#SBATCH -o slurm-%j.out

cd /work/young0nchc/go-board

export OMP_NUM_THREADS=4  # per process; workers also cap themselves

CKPT_DIR=checkpoints-19
RESUME=""
LATEST=$(ls -t "$CKPT_DIR"/iter_*.pt 2>/dev/null | head -1)
[ -n "$LATEST" ] && RESUME="--resume $LATEST"

exec .venv/bin/python -m train.train \
    --board-size 19 --komi 7.5 --komi-jitter 2.0 \
    --channels 128 --blocks 10 \
    --iterations 3000 \
    --games-per-iter 512 --parallel-games 96 \
    --simulations 600 --cheap-simulations 100 --full-search-prob 0.25 \
    --temperature-moves 30 --noise-fraction 0.3 \
    --steps-per-iter 1000 --batch-size 512 \
    --buffer-size 1500000 \
    --workers 8 \
    --checkpoint-dir "$CKPT_DIR" \
    $RESUME
