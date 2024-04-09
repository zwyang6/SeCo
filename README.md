# [CVPR2024] Separate and Conquer: Decoupling Co-occurrence via Decomposition and Representation for Weakly Supervised Semantic Segmentation [![arXiv](https://img.shields.io/badge/arXiv-2303.02506-b31b1b.svg)](http://arxiv.org/abs/2402.18467)


## News

* **If you find this work helpful, don't hesitate to give us a :star2: !**
* **` Mar. 21st, 2024`:** All Codes, logs, and checkpoints are available now🔥🔥🔥
* If you have any questions, please feel free to leave issues or contact us by zwyang21@m.fudan.edu.cn.

## Overview
We proposed a Separate and Conquer philosophy to effectively tackle the co-occurrence issue in WSSS. 
<p align="middle">
<img src="/sources/main_fig.png" alt="SeCo pipeline" width="1200px">
</p>


## Data Preparation

### PASCAL VOC 2012

#### 1. Download

``` bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
#### 2. Segmentation Labels

The augmented annotations are from [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html). The download link of the augmented annotations at
[DropBox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). After downloading ` SegmentationClassAug.zip `, you should unzip it and move it to `VOCdevkit/VOC2012/`. 

``` bash
VOCdevkit/
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── SegmentationObject
```

### MSCOCO 2014

#### 1. Download
``` bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
```

#### 2. Segmentation Labels

To generate VOC style segmentation labels for COCO, you could use the scripts provided at this [repo](https://github.com/alicranck/coco2voc), or just download the generated masks from [Google Drive](https://drive.google.com/file/d/147kbmwiXUnd2dW9_j8L5L0qwFYHUcP9I/view?usp=share_link).

``` bash
COCO/
├── JPEGImages
│    ├── train2014
│    └── val2014
└── SegmentationClass
     ├── train2014
     └── val2014
```

## Requirement

Please refer to the requirements.txt. 

We incorporate a regularization loss for segmentation. Please refer to the instruction for this [python extension](https://github.com/meng-tang/rloss/tree/master/pytorch#build-python-extension-module).

## Train SeCo
``` bash
### train voc
bash run_train.sh scripts/train_voc.py [gpu_number] [master_port] [gpu_device] train_voc

### train coco
bash run_train.sh scripts/train_coco.py [gpu_numbers] [master_port] [gpu_devices] train_coco
```

## Evaluate SeCo
``` bash
### eval voc
bash run_evaluate_seg_voc.sh tools/infer_seg_voc.py [gpu_device] [checkpoint_path]

### eval coco
bash run_evaluate_seg_coco.sh tools/infer_seg_coco.py [gpu_number] [master_port] [gpu_device] [checkpoint_path]
```

## Main Results
Semantic performance on VOC and COCO. Logs and weights are available now.
| Dataset | Backbone |  Val  | Test | Log | Weight |
|:-------:|:--------:|:-----:|:----:|:---:|:------:|
|   PASCAL VOC   |   ViT-B  | 74.0  | 73.8 | [log](logs/voc_train.log) | [weight](https://drive.google.com/file/d/1m5Yezcs1EPUuyJq1U_W0WuyPNj2Me4wT/view?usp=sharing)       |
|   MS COCO  |   ViT-B  |  46.7 |   -  | [log](logs/coco_train.log) | [weight](https://drive.google.com/file/d/1XpazzVBmSMwFsa7ei_Av22PGKA7pq7V0/view?usp=sharing)       |

## Citation 
Please cite our work if you find it helpful to your reseach. :two_hearts:
```bash
@article{yang2024separate,
  title={Separate and Conquer: Decoupling Co-occurrence via Decomposition and Representation for Weakly Supervised Semantic Segmentation},
  author={Yang, Zhiwei and Fu, Kexue and Duan, Minghong and Qu, Linhao and Wang, Shuo and Song, Zhijian},
  journal={arXiv preprint arXiv:2402.18467},
  year={2024}
}
```


## Acknowledgement
This repo is built upon [ToCo](https://github.com/rulixiang/ToCo), [DINO](https://github.com/facebookresearch/dino), and [SupCon](https://github.com/HobbitLong/SupContrast.git). Many thanks to their brilliant works!!!
