#!/usr/bin/env bash         \

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

PARTITION='dsta'
JOB_NAME=test
CONFIG=experiments/LoRA/ViT-B_prompt_lora_8.yaml
GPUS=1
INPUT_DIR=$2
WEIGHT_DECAY=0.0001

GPUS_PER_NODE=1
CPUS_PER_TASK=5
SRUN_ARGS=${SRUN_ARGS:-""}

mkdir -p logs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \


for LR in 0.005
do 
    for DATASET in imagenet-adversarial imagenet-rendition imagenet-sketch imagenetv2
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
            python supernet_train_prompt.py --data-path=./data/${DATASET} --data-set=${DATASET}-FS --cfg=${CONFIG} --resume=${INPUT_DIR}/checkpoint.pth --output_dir=${INPUT_DIR}/test/${DATASET} --batch-size=64 --mode=retrain --epochs=100 --lr=${LR} --weight-decay=${WEIGHT_DECAY} --eval --launcher="slurm"\
            2>&1 | tee -a logs/${currenttime}-${DATASET}-${LR}-test.log > /dev/null & 
            echo -e "\033[32m[ Please check log: \"logs/${currenttime}-${DATASET}-${LR}-test.log\" for details. ]\033[0m"
    done
done

