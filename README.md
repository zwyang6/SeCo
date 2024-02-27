# Separate and Conquer: Decoupling Co-occurrence via Decomposition and Representation for Weakly Supervised Semantic Segmentation[CVPR2024]

We proposed a Separate and Conquer philosophy to effectively tackle the co-occurrence issue in WSSS. 

### News

* **` Feb. 27nd, 2024`:** We released our paper on Arxiv. Further details can be found in code and our updated [arXiv]().
  
* Code will be available very soon.

## Overview

<p align="middle">
<img src="/sources/main_fig.png" alt="WeakSAM pipeline" width="1200px">
</p>

Attributed to the frequent coupling of co-occurring objects and the limited supervision from image-level labels, the challenging co-occurrence problem is widely present and leads to false activation of objects in weakly supervised semantic segmentation (WSSS). In this work, we devise a 'Separate and Conquer' training paradigm SeCo to tackle the co-occurrence challenge from dimensions of image space and feature space. In the image space, we propose to 'separate' the co-occurring objects with image decomposition by subdividing images into patches. Importantly, we assign each patch a category tag from Class Activation Maps (CAMs), which helps to locally remove the co-context bias and guide the subsequent representation. In the feature space, we propose to 'conquer' the false activation by enhancing semantic representation with multi-granularity knowledge contrast. To this end, a dual-teacher-single-student architecture is designed and tag-guided contrast is conducted to guarantee the correctness of knowledge and further facilitate the discrepancy among co-occurring objects. We streamline the multi-staged WSSS pipeline end-to-end and tackle co-occurrence without external supervision. Extensive experiments are conducted, validating the efficiency of our method tackling co-occurrence and the superiority over previous single-staged and even multi-staged competitors on PASCAL VOC and MS COCO. 


## Main results
Semantic performance on VOC and COCO. logs and weights will be public soon.
| Dataset | Backbone |  val  | test | log | weight |
|:-------:|:--------:|:-----:|:----:|:---:|:------:|
|   VOC   |   ViT-B  | 74.0  | 73.9 |     |        |
|   COCO  |   ViT-B  |  46.7 |   -  |     |        |


If you have any question, please feel free to contact the author by zwyang21@m.fudan.edu.cn.
