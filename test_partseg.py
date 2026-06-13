"""
Evaluate a trained part-segmentation model and export colored point clouds
for visualization (e.g. with Open3D / MeshLab / CloudCompare).

Usage (run from the repo root, same config as training):
    python test_partseg.py                      # evaluate + export with defaults
    python test_partseg.py num_visual=50        # export 50 shapes
    python test_partseg.py model=Menghao         # pick the trained model

The trained weights are read from the Hydra run directory of the matching
model, i.e. log/partseg/<model.name>/best_model.pth (the same place
train_partseg.py writes them to).

Each exported shape produces two ASCII .ply files under
log/partseg/<model.name>/visual/:
    <idx>_<category>_pred.ply   predicted part labels (colored)
    <idx>_<category>_gt.ply     ground-truth part labels (colored)
"""
import os
import logging
import importlib

import numpy as np
import torch
import hydra
import omegaconf
from tqdm import tqdm

from dataset import PartNormalDataset

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

NUM_PART = 50


def part_palette(num_parts=NUM_PART):
    """Deterministic, evenly spread RGB colors (0-255) for each part label."""
    import colorsys
    colors = np.zeros((num_parts, 3), dtype=np.uint8)
    for i in range(num_parts):
        h = (i * 0.61803398875) % 1.0  # golden-ratio hue spacing
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        colors[i] = (int(r * 255), int(g * 255), int(b * 255))
    return colors


def save_ply(path, xyz, rgb):
    n = xyz.shape[0]
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex %d\n" % n)
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            f.write("%f %f %f %d %d %d\n" % (xyz[i, 0], xyz[i, 1], xyz[i, 2],
                                             rgb[i, 0], rgb[i, 1], rgb[i, 2]))


def to_categorical(y, num_classes, device):
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    return new_y.to(device)


@hydra.main(config_path='config', config_name='partseg')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    num_visual = int(getattr(args, 'num_visual', 20))
    visual_dir = getattr(args, 'visual_dir', 'visual')
    os.makedirs(visual_dir, exist_ok=True)

    root = hydra.utils.to_absolute_path('data/shapenetcore_partanno_segmentation_benchmark_v0_normal/')
    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    args.input_dim = (6 if args.normal else 3) + 16
    args.num_class = NUM_PART
    num_category = 16

    model_module = importlib.import_module('models.{}.model'.format(args.model.name))
    classifier = getattr(model_module, 'PointTransformerSeg')(args).to(device)

    ckpt_path = 'best_model.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "best_model.pth not found in %s. Train first with train_partseg.py, "
            "or run this script from the same Hydra run dir." % os.getcwd())
    checkpoint = torch.load(ckpt_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    logger.info('Loaded checkpoint %s (epoch %s)', os.path.abspath(ckpt_path), checkpoint.get('epoch'))

    palette = part_palette()
    exported = 0

    total_correct = 0
    total_seen = 0
    shape_ious = {cat: [] for cat in seg_classes.keys()}

    with torch.no_grad():
        for points, label, target in tqdm(testDataLoader, total=len(testDataLoader)):
            cur_batch_size, NUM_POINT, _ = points.size()
            xyz_np = points[:, :, 0:3].cpu().numpy()
            points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
            seg_pred = classifier(torch.cat([points, to_categorical(label, num_category, device).repeat(1, points.shape[1], 1)], -1))
            cur_pred_logits = seg_pred.cpu().data.numpy()
            target_np = target.cpu().data.numpy()
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target_np[i, 0]]
                logits = cur_pred_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            total_correct += np.sum(cur_pred_val == target_np)
            total_seen += cur_batch_size * NUM_POINT

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target_np[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if np.sum(segl == l) == 0 and np.sum(segp == l) == 0:
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

                if exported < num_visual:
                    save_ply(os.path.join(visual_dir, '%d_%s_pred.ply' % (exported, cat)),
                             xyz_np[i], palette[segp])
                    save_ply(os.path.join(visual_dir, '%d_%s_gt.ply' % (exported, cat)),
                             xyz_np[i], palette[segl])
                    exported += 1

    all_ious = [v for cat in shape_ious for v in shape_ious[cat]]
    class_ious = [np.mean(shape_ious[cat]) for cat in shape_ious if len(shape_ious[cat])]
    logger.info('Accuracy: %.5f', total_correct / float(total_seen))
    logger.info('Class avg mIoU: %.5f', np.mean(class_ious))
    logger.info('Instance avg mIoU: %.5f', np.mean(all_ious))
    logger.info('Exported %d colored shapes to %s', exported, os.path.abspath(visual_dir))


if __name__ == '__main__':
    main()
