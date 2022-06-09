#!/usr/bin/env bash         \

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

PARTITION='dsta'
JOB_NAME=DG-SEARCH
CONFIG=./experiments/NOAH/supernet/supernet-B_prompt.yaml
GPUS=1
LIMITS=$1

GPUS_PER_NODE=1
CPUS_PER_TASK=5
SRUN_ARGS=${SRUN_ARGS:-""}

mkdir -p logs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \


for DATASET in imagenet
do 
    for SHOT in 16 
    do 
        export MASTER_PORT=$((12000 + $RANDOM % 20000))
        srun -p ${PARTITION} \
            --job-name=${JOB_NAME}-${DATASET} \
            --gres=gpu:${GPUS_PER_NODE} \
            --ntasks=${GPUS} \
            --ntasks-per-node=${GPUS_PER_NODE} \
            --cpus-per-task=${CPUS_PER_TASK} \
            --kill-on-bad-exit=1 \
            ${SRUN_ARGS} \
            python evolution.py --data-path=./data/${DATASET} --data-set=${DATASET}-FS --cfg=${CONFIG} --output_dir=saves/few-shot_${DATASET}_shot-${SHOT}_seed-0_lr-0.0005_wd-0.0001/search --batch-size=64 --resume=saves/few-shot_${DATASET}_shot-${SHOT}_seed-0_lr-0.0005_wd-0.0001/checkpoint.pth --param-limits=${LIMITS} --max-epochs=15 --few-shot-seed=0 --few-shot-shot=${SHOT} --launcher="slurm"\
            2>&1 | tee -a logs/${currenttime}-${DATASET}-${SHOT}-dg-search.log > /dev/null & 
            echo -e "\033[32m[ Please check log: \"logs/${currenttime}-${DATASET}-${SHOT}-dg-search.log\" for details. ]\033[0m"
    done
done
