<div align="center">

<h1>Neural Prompt Search</h1>

<div>
    <a href='https://davidzhangyuanhan.github.io/' target='_blank'>Yuanhan Zhang</a>&emsp;
    <a href='https://kaiyangzhou.github.io/' target='_blank'>Kaiyang Zhou</a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu<sup>*</sup></a>
</div>
<div>
    S-Lab, Nanyang Technological University
</div>


<img src="figures/motivation.png">


    
The idea is simple: we view existing parameter-efficient tuning modules, including [Adapter](), [LoRA]() and [VPT](), as prompt modules and propose to search the optimal configuration via neural architecture search. Our approach is named **NOAH** (Neural prOmpt seArcH).

---

<p align="center">
  <a href="https://github.com/Davidzhangyuanhan/NOAH" target='_blank'>[arXiv]</a>
</p>

</div>



## Updatas
[05/2022] [arXiv](https://github.com/Davidzhangyuanhan/NOAH) paper has been **released**.

## Environment Setup
```
conda create -n NOAH python=3.8
conda activate NOAH
pip install -r requirements.txt
```

## Data Preparation

### 1. Visual Task Adaptation Benchmark (VTAB)
```
cd data/vtab-source
python get_vtab1k.py
```

### 2. Few-Shot and Domain Generation

- Images

    Please refer to [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to download the datasets.

- Train/Val/Test splits

    Please refer to files under `data/XXX/XXX/annotations` for the detail information.


## Quick Start For NOAH
We use the VTAB experiments as examples.

### 1. Downloading the Pre-trained Model
| Model | Link |
|-------|------|
|ViT B/16 | [link](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz)|

### 2. Supernet Training
```
sh configs/NOAH/VTAB/supernet/slurm_train_vtab.sh PATH-TO-YOUR-PRETRAINED-MODEL
```

### 3. Subnet Search
```
sh configs/NOAH/VTAB/search/slurm_search_vtab.sh
```
### 4. Subnet Retraining
```
sh configs/NOAH/VTAB/subnet/slurm_retrain_vtab.sh PATH-TO-YOUR-PRETRAINED-MODEL
```

## Citation
If you use this code in your research, please kindly cite this work.
```
@inproceedings{zhang2022NOAH,
      title={Neural Prompt Search}, 
      author={Yuanhan Zhang and Kaiyang Zhou and Ziwei Liu},
      year={2022},
      archivePrefix={arXiv},
}
```

## Acknoledgments
Part of the code is borrowed from [CoOp](https://github.com/KaiyangZhou/CoOp), [AutoFormer](https://github.com/microsoft/Cream/tree/main/AutoFormer), [timm](https://github.com/rwightman/pytorch-image-models) and [mmcv](https://github.com/open-mmlab/mmcv).

Thanks Zhou Cong (https://chongzhou96.github.io/) for the code of downloading the VTAB-1k.

