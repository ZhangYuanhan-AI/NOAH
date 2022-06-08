#!/usr/bin/env bash         \

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

PARTITION='dsta'
JOB_NAME=AD-DG
CONFIG=$1
GPUS=1
CKPT=$2
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
        for SEED in 0
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
                    python supernet_train_prompt.py --data-path=./data/${DATASET} --data-set=${DATASET}-FS --cfg=${CONFIG} --resume=${CKPT} --output_dir=./saves/few-shot_${DATASET}_shot-${SHOT}_seed-${SEED}_lr-${LR}_wd-${WEIGHT_DECAY}_adapter --batch-size=64 --lr=${LR} --epochs=100 --is_adapter --weight-decay=${WEIGHT_DECAY} --few-shot-seed=${SEED} --few-shot-shot=${SHOT} --launcher="slurm"\
                    2>&1 | tee -a logs/${currenttime}-${DATASET}-${LR}-seed-${SEED}-shot-${SHOT}-adpter.log > /dev/null & 
                    echo -e "\033[32m[ Please check log: \"logs/${currenttime}-${DATASET}-${LR}-seed-${SEED}-${SHOT}-shot-adpter.log\" for details. ]\033[0m"
          done
        done
    done
done
