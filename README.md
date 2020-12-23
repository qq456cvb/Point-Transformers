# Pytorch Implementation of Various Point Transformers

Recently, various methods applied transformers to point clouds: [PCT: Point Cloud Transformer (Meng-Hao Guo et al.)](https://arxiv.org/abs/2012.09688), [Point Transformer (Nico Engel et al.)](https://arxiv.org/abs/2011.00931), [Point Transformer (Hengshuang Zhao et al.)](https://arxiv.org/abs/2012.09164). This repo is a pytorch implementation for these methods and aims to compare them under a fair setting. Currently, Point Transformer (Nico Engel et al.) is implemented.


## Classification
### Data Preparation
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `modelnet40_normal_resampled`.

### Run
```
python train.py
```
### Results
TBA

### Miscellaneous
Some code and training settings are borrowed from https://github.com/yanx27/Pointnet_Pointnet2_pytorch.
## TODOs
- [ ] implement Point Transformer (Hengshuang Zhao et al.)
- [ ] implement PCT: Point Cloud Transformer (Meng-Hao Guo et al.)