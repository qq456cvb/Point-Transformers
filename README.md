# Point Transformers (PyTorch)

A PyTorch implementation and fair comparison of three transformer architectures for point clouds:

- [Point Transformer](https://arxiv.org/abs/2012.09164) — Hengshuang Zhao et al.
- [PCT: Point Cloud Transformer](https://arxiv.org/abs/2012.09688) — Meng-Hao Guo et al.
- [Point Transformer](https://arxiv.org/abs/2011.00931) — Nico Engel et al.

All three models are implemented behind a common training pipeline (same data, same augmentation, same schedule), so their results can be compared under one consistent setting. Configuration is managed with [Hydra](https://hydra.cc/), so switching models or hyperparameters is a one-flag change.

## Project Structure

```
config/
├── cls.yaml              # classification hyperparameters
├── partseg.yaml          # part segmentation hyperparameters
└── model/                # per-model configs: Hengshuang / Menghao / Nico
models/
├── Hengshuang/           # Point Transformer (vector attention + transition down/up)
├── Menghao/              # PCT: Point Cloud Transformer
└── Nico/                 # Point Transformer (SortNet + local-global attention)
train_cls.py              # ModelNet40 classification training/eval
train_partseg.py          # ShapeNet part segmentation training/eval
dataset.py                # ModelNet40 / ShapeNetPart data loaders
provider.py               # point cloud augmentations
```

## Installation

```bash
pip install -r requirements.txt
```

Requires PyTorch with CUDA; the training scripts assume a GPU.

## Classification (ModelNet40)

### Data

Download the resampled, aligned **ModelNet40** ([modelnet40_normal_resampled.zip](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)) and extract it to `modelnet40_normal_resampled/` at the repo root.

### Train

```bash
# default model is set in config/cls.yaml
python train_cls.py

# or pick a model explicitly
python train_cls.py model=Hengshuang
python train_cls.py model=Menghao
python train_cls.py model=Nico

# sweep all three with Hydra multirun
python train_cls.py model=Hengshuang,Menghao,Nico -m
```

Logs and the best checkpoint (`best_model.pth`) are written to `log/cls/<model>/`.

### Results

Adam, learning rate decay 0.3 every 50 epochs, 200 epochs total; data augmentation follows [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). For Hengshuang and Nico the initial LR is 1e-3 (these hyperparameters could likely be tuned further); for Menghao it is 1e-4, as suggested by the [author](https://github.com/MenghaoGuo).

ModelNet40 classification accuracy (instance average):

| Model | Accuracy |
| -- | -- |
| Hengshuang | 91.7 |
| Menghao | 92.6 |
| Nico | 85.5 |

## Part Segmentation (ShapeNetPart)

### Data

Download the aligned **ShapeNetPart** benchmark ([shapenetcore_partanno_segmentation_benchmark_v0_normal.zip](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)) and extract it to `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`.

### Train

```bash
python train_partseg.py model=Hengshuang
```

Logs and checkpoints are written to `log/partseg/<model>/`. Currently only Hengshuang's architecture has a segmentation head implemented.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- Training pipeline and data augmentation adapted from [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).
- The Menghao (PCT) implementation is adapted from the author's Jittor version: [MenghaoGuo/PCT](https://github.com/MenghaoGuo/PCT).
