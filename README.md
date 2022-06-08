# Neural Prompt Search
![fig1](figures/motivation.png)
We propose to search, instead of hand-engineering, prompt modules for parameter-efficient transfer learning.

## Updatas
[03/2022] [arXiv](https://github.com/Davidzhangyuanhan/NOAH) paper has been **released**.

## Environment Setup
```
conda create -n NOAH python=3.8
conda activate NOAH
pip install -r requirements.txt
```

## Data Preparation

### Visual Task Adaptation Benchmark (VTAB)
TBA

### Few-Shot Setting and Domain Generation Setting

- Images

    Please refer to [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to download the datasets.

- Train/Val/Test splits

    Please refer to files under `data/XXX/XXX/annotations` for the detail information.


## Quick Start
### Downloading the Pre-trained Model
| Model | Link |
|-------|------|
|ViT B/16 | [link](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz)|

### Supernet training

### Subnet Search

### Subnet Retraining

### Subnet Evaluation

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
The codes are inspired by [CoOp](https://github.com/KaiyangZhou/CoOp), [AutoFormer](https://github.com/microsoft/Cream/tree/main/AutoFormer), [timm](https://github.com/rwightman/pytorch-image-models) and [mmcv](https://github.com/open-mmlab/mmcv).

