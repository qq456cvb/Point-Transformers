# Pytorch Implementation of Various Point Transformers

Recently, various methods applied transformers to point clouds: [PCT: Point Cloud Transformer (Meng-Hao Guo et al.)](https://arxiv.org/abs/2012.09688), [Point Transformer (Nico Engel et al.)](https://arxiv.org/abs/2011.00931), [Point Transformer (Hengshuang Zhao et al.)](https://arxiv.org/abs/2012.09164). This repo is a pytorch implementation for these methods and aims to compare them under a fair setting. Currently, all three methods are implemented, while tuning their hyperparameters.


## Classification
### Data Preparation
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `modelnet40_normal_resampled`.

### Run
Change which method to use in `config/cls.yaml` and run
```
python train_cls.py
```
### Results
Using Adam with learning rate decay 0.3 for every 50 epochs, train for 200 epochs; data augmentation follows [this repo](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). For Hengshuang and Nico, initial LR is 1e-3 (I would appreciate if someone could fine-tune these hyper-paramters); for Menghao, initial LR is 1e-4, as suggested by the [author](https://github.com/MenghaoGuo). ModelNet40 classification results (instance average) are listed below:
| Model | Accuracy |
|--|--|
| Hengshuang | 91.7 |
| Menghao | 92.6 |
| Nico |  85.5 |


## Part Segmentation
### Data Preparation
Download alignment **ShapeNet** [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) and save in `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`.

### Run
Change which method to use in `config/partseg.yaml` and run
```
python train_partseg.py
```
### Results
Currently only Hengshuang's method is implemented.

### Miscellaneous
Some code and training settings are borrowed from https://github.com/yanx27/Pointnet_Pointnet2_pytorch.
Code for [PCT: Point Cloud Transformer (Meng-Hao Guo et al.)](https://arxiv.org/abs/2012.09688) is adapted from the author's Jittor implementation https://github.com/MenghaoGuo/PCT.

