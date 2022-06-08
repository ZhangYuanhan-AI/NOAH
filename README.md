# Neural Prompt Search
![fig1](figures/motivation.png)
We propose to search, instead of hand-engineering, prompt modules for parameter-efficient transfer learning.

## Updatas
[05/2022] [arXiv](https://github.com/Davidzhangyuanhan/NOAH) paper has been **released**.

## Environment Setup
```
conda create -n NOAH python=3.8
conda activate NOAH
pip install -r requirements.txt
```

## Data Preparation

### Visual Task Adaptation Benchmark (VTAB)
```
cd data/vtab-source
python get_vtab1k.py
```

### Few-Shot Setting and Domain Generation Setting

- Images

    Please refer to [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to download the datasets.

- Train/Val/Test splits

    Please refer to files under `data/XXX/XXX/annotations` for the detail information.


## Quick Start For NOAH
We use the VTAB experiments as examples.

### Downloading the Pre-trained Model
| Model | Link |
|-------|------|
|ViT B/16 | [link](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz)|

### Supernet training
```
sh configs/NOAH/VTAB/supernet/slurm_train_vtab.sh PATH-TO-YOUR-PRETRAINED-MODEL
```

### Subnet Search
```
sh configs/NOAH/VTAB/search/slurm_search_vtab.sh
```
### Subnet Retraining
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
The codes are inspired by [CoOp](https://github.com/KaiyangZhou/CoOp), [AutoFormer](https://github.com/microsoft/Cream/tree/main/AutoFormer),[timm](https://github.com/rwightman/pytorch-image-models) and [mmcv](https://github.com/open-mmlab/mmcv).

Thanks Zhou Cong (https://chongzhou96.github.io/) for the code for downloading the VTAB-1k.

