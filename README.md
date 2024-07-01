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
- [License](#license)
- [Contact](#contact)

## Introduction

## Requirements
- Python 3.8
- PyTorch 2.1.0
- torchvision 0.16.0
- CUDA 11.8

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/efkandurakli/DGMaskRCNN.git
   cd DGMaskRCNN
   
2. Create a conda virtual environment and activate it
   ```
   conda create --name dgmaskrcnn python=3.8
   conda activate dgmaskrcnn
   
3. Install the required packages listed in requirements section


## Dataset
We employed [IEEE DFC 2023 Dataset](https://www.grss-ieee.org/community/technical-committees/2023-ieee-grss-data-fusion-contest/). Download the dataset and organize it as follows:
```md
datasets/DFC2023/track1/
├── annottations/
│   ├── buildingonly/
│   |   ├── train.json
|   |   ├── test.json
|   |   ├── val.json
├── images/
```

## Usage

### Training




### Inference

