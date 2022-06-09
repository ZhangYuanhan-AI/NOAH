#!/usr/bin/env bash         \

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

PARTITION='dsta'
JOB_NAME=VTAB-SEARCH
CONFIG=./experiments/NOAH/supernet/supernet-B_prompt.yaml
GPUS=1
LIMITS=$1

GPUS_PER_NODE=1
CPUS_PER_TASK=5
SRUN_ARGS=${SRUN_ARGS:-""}


mkdir -p logs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \


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
        ${SRUN_ARGS} \
        python evolution.py --data-path=./data/vtab-1k/${DATASET} --data-set=${DATASET} --cfg=${CONFIG} --output_dir=saves/${DATASET}_supernet_lr-0.0005_wd-0.0001/search_limit-${LIMITS} --batch-size=64 --resume=saves/${DATASET}_supernet_lr-0.0005_wd-0.0001/checkpoint.pth --param-limits=${LIMITS} --max-epochs=15 --no_aug --inception --direct_resize --mixup=0 --cutmix=0 --smoothing=0 --launcher="slurm"\
        2>&1 | tee -a logs/${currenttime}-${DATASET}-vtab-search.log > /dev/null & 
        echo -e "\033[32m[ Please check log: \"logs/${currenttime}-${DATASET}-vtab-search.log\" for details. ]\033[0m"
done
# done
