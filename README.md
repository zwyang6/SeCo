# [CVPR2024] Separate and Conquer: Decoupling Co-occurrence via Decomposition and Representation for Weakly Supervised Semantic Segmentation

We proposed a Separate and Conquer philosophy to effectively tackle the co-occurrence issue in WSSS. 

## News

* **` Feb. 29th, 2024`:** We released our paper on Arxiv. Further details can be found in the updated [arXiv](http://arxiv.org/abs/2402.18467).
  
* **` Mar. 1st, 2024`:** Code for PASCAL VOC has been available.
* **`If you find this work helpful, please give us a :star2: to receive the updation !`**

## Overview

<p align="middle">
<img src="/sources/main_fig.png" alt="SeCo pipeline" width="1200px">
</p>

Attributed to the frequent coupling of co-occurring objects and the limited supervision from image-level labels, the challenging co-occurrence problem is widely present and leads to false activation of objects in weakly supervised semantic segmentation (WSSS). In this work, we devise a 'Separate and Conquer' scheme SeCo to tackle this issue from dimensions of image space and feature space. In the image space, we propose to 'separate' the co-occurring objects with image decomposition by subdividing images into patches. Importantly, we assign each patch a category tag from Class Activation Maps (CAMs), which spatially helps remove the co-context bias and guide the subsequent representation. In the feature space, we propose to 'conquer' the false activation by enhancing semantic representation with multi-granularity knowledge contrast. To this end, a dual-teacher-single-student architecture is designed and tag-guided contrast is conducted to guarantee the correctness of knowledge and further facilitate the discrepancy among co-occurring objects. We streamline the multi-staged WSSS pipeline end-to-end and tackle co-occurrence without external supervision. Extensive experiments are conducted, validating the efficiency of our method tackling co-occurrence and the superiority over previous single-staged and even multi-staged competitors on PASCAL VOC and MS COCO.


## Data Preparation

### PASCAL VOC 2012

#### 1. Download

``` bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
#### 2. Segmentation Labels

The augmented annotations are from [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html). Here is a download link of the augmented annotations at
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

Please refer to requirements.txt

## Train SeCo
``` bash
### train voc
bash run_voc.sh scripts/train_voc.py [master_port] [gpu_device] train_voc

### train coco
bash run_coco.sh scripts/train_coco.py [master_port] [gpu_devices] train_coco
```

## Train SeCo
``` bash
### eval voc
bash run_evaluate_seg_voc.sh tools/infer_seg_voc.py [gpu_device] [checkpoint_path]

### train coco
bash run_evaluate_seg_voc.sh tools/infer_seg_coco.py [gpu_device] [checkpoint_path]
```

## Main results
Semantic performance on VOC and COCO. logs and weights will be public soon.
| Dataset | Backbone |  val  | test | log | weight |
|:-------:|:--------:|:-----:|:----:|:---:|:------:|
|   VOC   |   ViT-B  | 74.0  | 73.9 |     |        |
|   COCO  |   ViT-B  |  46.7 |   -  |     |        |


If you have any question, please feel free to contact the author by zwyang21@m.fudan.edu.cn.

## Acknowledgement
This repo is built upon [ToCo](https://github.com/rulixiang/ToCo), [DINO](https://github.com/facebookresearch/dino), and [SupCon](https://github.com/HobbitLong/SupContrast.git). Many thanks to their brilliant works!!!
