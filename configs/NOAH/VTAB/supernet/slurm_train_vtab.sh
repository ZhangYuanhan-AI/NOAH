#!/usr/bin/env bash         \

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

PARTITION='dsta'
JOB_NAME=VTAB-SUPERNET
CONFIG=./experiments/NOAH/supernet/supernet-B_prompt.yaml
GPUS=1
CKPT=$1
WEIGHT_DECAY=0.0001

GPUS_PER_NODE=1
CPUS_PER_TASK=5
SRUN_ARGS=${SRUN_ARGS:-""}

mkdir -p logs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \ 

for LR in 0.0005
do 
    for DATASET in cifar100 caltech101 dtd oxford_flowers102 svhn sun397 oxford_pet patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do 
        export MASTER_PORT=$((12000 + $RANDOM % 20000))
        srun -p ${PARTITION} \
            --job-name=${JOB_NAME}-${DATASET} \
            --gres=gpu:${GPUS_PER_NODE} \
            --ntasks=${GPUS} \
            --ntasks-per-node=${GPUS_PER_NODE} \
            --cpus-per-task=${CPUS_PER_TASK} \
            --kill-on-bad-exit=1 \
            -w SG-IDC1-10-51-2-73 \
            ${SRUN_ARGS} \
            python supernet_train_prompt.py --data-path=./data/vtab-1k/${DATASET} --data-set=${DATASET} --cfg=${CONFIG} --resume=${CKPT} --output_dir=./saves/${DATASET}_supernet_lr-${LR}_wd-${WEIGHT_DECAY} --batch-size=64 --lr=${LR} --epochs=500 --weight-decay=${WEIGHT_DECAY} --no_aug --direct_resize --mixup=0 --cutmix=0 --smoothing=0 --launcher="slurm"\
            2>&1 | tee -a logs/${currenttime}-${DATASET}-${LR}-vtab-supernet.log > /dev/null & 
            echo -e "\033[32m[ Please check log: \"logs/${currenttime}-${DATASET}-${LR}-vtab-supernet.log\" for details. ]\033[0m"
    done
done