# A Domain Generalized Mask R-CNN for Building Instance Segmentation
This repository contains the code for the paper titled "A Domain Generalized Mask R-CNN for Building Instance Segmentation" accepted for IGARSS 2024.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
  
## Introduction

### **Abstract** 
> *Building instance segmentation is a critical task for urban planning. It has been extensively studied through state-of-the-art instance segmentation models, and great advances have been reported. However, the issue of domain shift referring to disparities between training and target data distributions remains elusive. Although domain adaptation can help tackle it, it still requires access to target domain data. In this paper, we explore the problem of building extraction in the domain generalization setting, where no access to either target labels or target data is assumed. Mask R-CNN is equipped with image, instance, and pixel-level domain adversarial modules in order to encourage the extraction of domain-invariant features. The preliminary results obtained with cross-continent domain generalization are promising.*

### **Architecture Overview** 

![Overview](/images/figure.png)

## Requirements
- Python 3.11
- PyTorch 2.1.0
- torchvision 0.16.0
- lightning 2.0.9
- CUDA 11.8

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/efkandurakli/DGMaskRCNN.git
   cd DGMaskRCNN
   ```
   
2. Create a conda virtual environment and activate it
   ```
   conda create --name dgmaskrcnn python=3.11
   conda activate dgmaskrcnn
   ```
   
3. Install the required packages listed in requirements section


## Dataset
We employed [IEEE DFC 2023 Dataset](https://www.grss-ieee.org/community/technical-committees/2023-ieee-grss-data-fusion-contest/). Download the dataset and organize it based on the Table 1 in the paper as follows:
```md
datasets/DFC2023/track1/
├── annottations/
│   ├── buildingonly/
│       ├── train.json
|       ├── test.json
|       ├── val.json
├── images/
```

## Usage

### Training

#### Baseline (Mask R-CNN)
```
python train.py
```
#### With image-, instance, and pixel-level domain alignment
```
python train.py --image-dg --box-dg --mask-dg --reg_weights 0.1 0.1 0.1
```


### Inference
```
python test.py --checkpoint-dir $CHECKPOINT_DIR --checkpoint-file-name $CHECKPOINT_FILENAME
```

## Results

|               | image   | instance |     pixel   | mAP     | mAP@50   | 
|---------------|:-------:|:--------:|:-----------:|:-------:|:--------:|
| Mask R-CNN    |         |          |             |  0.113  |  0.238   |
| DGMaskRCNN    |   ✓     |          |             |  **0.120**  |  **0.254**   | 
| DGMaskRCNN    |         |     ✓    |             |  0.119  |  0.252   |
| DGMaskRCNN    |         |          |       ✓     |  0.117  |  0.242   |
| DGMaskRCNN    |   ✓     |     ✓    |       ✓     |  0.114  |  0.246   |

## Citation
```
@inproceedings{durakkli2024,
  title={A Domain Generalized Mask R-CNN for Building Instance Segmentation},
  author={E.~Durakli and P.~Bosilj and C.~Fox and E.Aptoula},
  booktitle={IGARSS 2024 IEEE International Geoscience and Remote Sensing Symposium},
  year={2024}
}
```
## Acknowledgement

This study has been supported by the Sabanci University Project number: B.A.CF-23-02672.
