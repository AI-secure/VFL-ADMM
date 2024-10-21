# Improving Privacy-Preserving Vertical Federated Learning by Efficient Communication with ADMM

This is the official repository for our paper "Improving Privacy-Preserving Vertical Federated Learning by Efficient Communication with ADMM" (SaTML 2024)

## Contents
- [Introduction](#introduction)
- [Install](#install)
- [Dataset](#dataset)
- [Run](#run)


## Introduction
This repository contains the code for our SaTML 2024 [Improving Privacy-Preserving Vertical Federated Learning by Efficient Communication with ADMM](https://openreview.net/forum?id=Xu10VyVnSE). 
Federated learning (FL) enables distributed resource-constrained devices to jointly train shared models while keeping the training data local for privacy purposes. Vertical FL (VFL), which allows each client to collect partial features, has attracted intensive research efforts recently. We identified the main challenges that existing VFL frameworks are facing: the server needs to communicate gradients with the clients for each training step, incurring high communication cost that leads to rapid consumption of privacy budgets. To address these challenges, in this paper, we introduce a VFL framework with multiple heads (VIM), which takes the separate contribution of each client into account, and enables an efficient decomposition of the VFL optimization objective to sub-objectives that can be iteratively tackled by the server and the clients on their own. In particular, we propose an Alternating Direction Method of Multipliers (ADMM)-based method to solve our optimization problem, which allows clients to conduct multiple local updates before communication, and thus reduces the communication cost and leads to better performance under differential privacy.

## Install

Create a conda environment and install additional packages:

```bash
conda create -n vfladmm python=3.8
conda activate vfladmm
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```


## Dataset

Prepare the datasets in the folder `raw_data`. We support MNIST, CIFAR10, ModelNet40, NUS-WIDE. 

- MNIST, CIFAR10 will be downloaded automatically. 
- ModelNet40: a multi-view image dataset, containing the shaded images from different views for the same object. Download it as follows: 
```bash 
cd raw_data
wget http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz
tar -xvf shaded_images.tar.gz
```
- NUS-WIDE:  a multi-modality dataset with 634 low-level image features and 1000 textual tag features. Download them from the [original paper](https://dl.acm.org/doi/10.1145/1646396.1646452). 




## Run

To run different Vertical FL methods on datasets, please use:

```Shell
export CUDA_VISIBLE_DEVICES=0
sh scripts/clean/{dataset}/{method}.sh
```

-  `dataset` can be `mnist`,`cifar`,`modelnet`,`nus`
- Vertical FL `method` with model splitting setting: `admm`, `fedbcd`, `splitlearn`, `vafl`.
- Vertical FL `method` without model splitting setting: `admmjoint`, `fdml`.

Note: please add `--wandb_key {YOUR_WANDB_KEY}` in the script if wandb visualization  `--vis` is enabled. 


## <a name="Citation"></a> Citation

If you find this work useful for your research and applications, please cite using this BibTeX:
```bibtex
@inproceedings{
xie2024improving,
title={Improving Privacy-Preserving Vertical Federated Learning by Efficient Communication with {ADMM}},
author={Chulin Xie and Pin-Yu Chen and Qinbin Li and Arash Nourian and Ce Zhang and Bo Li},
booktitle={2nd IEEE Conference on Secure and Trustworthy Machine Learning},
year={2024},
url={https://openreview.net/forum?id=Xu10VyVnSE}
}
```

## Questions
If you have any questions related to the code or the paper, feel free to email Chulin (chulinx2@illinois.edu) or open an issue.


