#!/usr/bin/env bash         \

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

PARTITION='dsta'
JOB_NAME=DG-RT
GPUS=1
CKPT=$1
WEIGHT_DECAY=0.0001

GPUS_PER_NODE=1
CPUS_PER_TASK=5
SRUN_ARGS=${SRUN_ARGS:-""}


mkdir -p logs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

for LR in 0.005 
do 
    for DATASET in imagenet
    do 
        for SHOT in 16
        do
            for SEED in 0 1 2 
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
                    python supernet_train_prompt.py --data-path=./data/${DATASET} --data-set=${DATASET}-FS --cfg=experiments/NOAH/subnet/few-shot/ViT-B_prompt_${DATASET}_shot${SHOT}-seed0.yaml --resume=${CKPT} --output_dir=saves/few-shot_${DATASET}_shot-${SHOT}_seed-0_lr-0.0005_wd-0.0001/retrain_lr-${LR}-seed-${SEED} --batch-size=64 --mode=retrain --epochs=100 --lr=${LR} --weight-decay=${WEIGHT_DECAY} --few-shot-seed=0 --few-shot-shot=${SHOT} --few-shot-seed=${SEED} --launcher="slurm"\
                    2>&1 | tee -a logs/${currenttime}-${DATASET}-lr-${LR}-shot-${SHOT}-seed-${SEED}-dg-rt.log > /dev/null & 
                    echo -e "\033[32m[ Please check log: \"logs/${currenttime}-${DATASET}-lr-${LR}-shot-${SHOT}-seed-${SEED}-dg-rt.log\" for details. ]\033[0m"
            done
        done
    done
done