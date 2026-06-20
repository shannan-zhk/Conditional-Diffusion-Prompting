import os
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import random
from utils import (
    generalized_energy_distance_iou, hm_iou_cal, dice_max_cal2
)
from lidc_dataset import LIDC_IDRI, RandomGenerator
from asam import ASAM

# Configure logging
logging.basicConfig(filename='evaluation_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate the model with specified epochs and weights.')
parser.add_argument('--epochs', nargs='+', type=int, default=[10, 20, 30, 50], help='Epochs to load weights from.')
parser.add_argument('--combined_weights_path', type=str, default='checkpoint/final_weights.pth', help='Path to the combined weights file.')
parser.add_argument('--gpuid', type=int, default=0, help='ID of the GPU to use.')
parser.add_argument('--batch_size', type=int, default=64,help='Batch size for data loading.',)
parser.add_argument('--total_samples', type=int, default=20,help='Total number of samples to generate.')
parser.add_argument('--dataset', type=str, default='lidc', choices=['lidc', 'qubiq', 'isbi', 'isic'], help='Choose dataset: lidc, qubiq, isbi, or isic')
parser.add_argument('--qubiq_val_dir', type=str, default='QUBIQ/validation_data_qubiq2021_QC', help='QUBIQ validation dir root')
parser.add_argument('--qubiq_task', type=str, default=None, help='QUBIQ task name, e.g., prostate, brain-growth, brain-tumor, pancreas, pancreatic-lesion')
parser.add_argument('--isbi_image_dir', type=str, default='/home/u2024111264/share/zhk/asam原代码/ISBI/ISBI2016_ISIC_Part1_Training_Data', help='ISBI images directory')
parser.add_argument('--isbi_mask_dir', type=str, default='/home/u2024111264/share/zhk/asam原代码/ISBI/ISBI2016_ISIC_Part1_Training_GroundTruth', help='ISBI masks directory')
parser.add_argument('--isic_dir', type=str, default='/home/u2024111264/share/zhk/GTRasam/GTR/data/isic256_3_style', help='ISIC dataset root directory (Images/ and Segmentations/ subfolders)')
parser.add_argument(
    '--gtr_ckpt_path',
    type=str,
    default='/home/u2024111264/share/zhk/GTRasamdiffusion/GTR/saved_models/isic3/2025_12_22_13_44/best_model.pt',
    # default='/home/u2024111264/share/zhk/GTRasamdiffusion/GTR/saved_models/isic3/2025_12_08_14_01/best_model.pt',
    help='(可选) 训练好的 GTR checkpoint 路径，用作 ASAM 的冻结先验分支；LIDC+GTR 评估时需要与训练保持一致',
)
args = parser.parse_args()

# Set device
device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')

# Load combined weights
combined_weights = torch.load(args.combined_weights_path, map_location=device)

# Initialize networks
def initialize_networks(epochs):
    networks = []
    for epoch in epochs:
        epoch_key = f'epoch_{epoch}'
        if epoch_key in combined_weights:
            # 与训练阶段保持同一结构：如果提供了 GTR ckpt，则在 ASAM 中挂载冻结 GTR 分支。
            # 同时将当前评估的数据集名称传入 ASAM，用于选择合适的 GTR 输入通道数。
            net = ASAM(
                gtr_ckpt_path=args.gtr_ckpt_path,
                dataset=args.dataset,
            ).to(device)
            net.load_state_dict(combined_weights[epoch_key]['model_state_dict'])
            net.eval()
            networks.append((net, combined_weights[epoch_key]['mask_weights'].to(device)))
        else:
            print(f"Warning: Weights for epoch {epoch} not found.")
    return networks

def self_similarity_cal(predictions, eps=1e-7):
    """Average pairwise Dice similarity among generated masks; lower means more diverse."""
    num_preds = predictions.shape[0]
    if num_preds < 2:
        return 1.0

    pred_flat = predictions.float().reshape(num_preds, -1)
    intersection = pred_flat @ pred_flat.t()
    volumes = pred_flat.sum(dim=1, keepdim=True)
    dice_matrix = (2.0 * intersection + eps) / (volumes + volumes.t() + eps)

    pair_mask = torch.triu(torch.ones(num_preds, num_preds, dtype=torch.bool, device=predictions.device), diagonal=1)
    return dice_matrix[pair_mask].mean().item()

def bhattacharyya_distance_cal(predictions, eps=1e-7):
    """Average pairwise Bhattacharyya distance between normalized foreground maps."""
    num_preds = predictions.shape[0]
    if num_preds < 2:
        return 0.0

    pred_flat = predictions.float().reshape(num_preds, -1)
    pred_dist = pred_flat + eps
    pred_dist = pred_dist / pred_dist.sum(dim=1, keepdim=True)

    coeff_matrix = torch.sqrt(pred_dist[:, None, :] * pred_dist[None, :, :]).sum(dim=-1)
    distance_matrix = -torch.log(torch.clamp(coeff_matrix, min=eps, max=1.0))

    pair_mask = torch.triu(torch.ones(num_preds, num_preds, dtype=torch.bool, device=predictions.device), diagonal=1)
    return distance_matrix[pair_mask].mean().item()

ged_score = dice_max2_score = hm_iou_score = 0
Dxy_score = Dxx_score = Dyy_score = 0.0
# Track decomposed GED terms
Dxy_score = Dxx_score = Dyy_score = 0.0
self_similarity_score = bhattacharyya_distance_score = 0.0
networks = initialize_networks(args.epochs)

# Log the weights being used
logging.info(f"Using combined weights from {args.combined_weights_path} for epochs: {args.epochs}")

import math

# Prepare dataset
if args.dataset == 'lidc':
    db = LIDC_IDRI(dataset_location='LIDC/data/', transform=transforms.Compose([
        RandomGenerator(output_size=[128, 128])
    ]))
elif args.dataset == 'qubiq':
    from qubiq_dataset import QUBIQ_Dataset
    val_base = args.qubiq_val_dir if args.qubiq_task is None else os.path.join(args.qubiq_val_dir, args.qubiq_task)
    db = QUBIQ_Dataset(
        base_dir=val_base,
        target_size=(128, 128),
        raters_per_sample=None,
        task_name=args.qubiq_task,
        transform=transforms.Compose([RandomGenerator(output_size=[128, 128])])
    )
elif args.dataset == 'isbi':
    from isbi_dataset import ISBI_Dataset
    if args.isbi_image_dir is None or args.isbi_mask_dir is None:
        raise ValueError('For ISBI dataset, --isbi_image_dir and --isbi_mask_dir must be provided')
    db = ISBI_Dataset(
        image_dir=args.isbi_image_dir,
        mask_dir=args.isbi_mask_dir,
        target_size=(128, 128),
        transform=transforms.Compose([RandomGenerator(output_size=[128, 128])])
    )
elif args.dataset == 'isic':
    from isic_dataset import ISIC_Dataset
    db = ISIC_Dataset(
        base_dir=args.isic_dir,
        target_size=(128, 128),
    )
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")
dataset_size = len(db)
indices = list(range(dataset_size))
train_split = int(np.floor(0.6 * dataset_size))
validation_split = int(np.floor(0.8 * dataset_size))
test_indices = indices[validation_split:]

if args.dataset == 'lidc' or args.dataset == 'isic':
    test_dataset = Subset(db, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
elif args.dataset == 'qubiq':
    # QUBIQ: directly evaluate on the provided validation set
    test_dataset = db
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
elif args.dataset == 'isbi':
    # ISBI: use 80% for training, 20% for validation (same as training script)
    indices = list(range(dataset_size))
    train_split = int(np.floor(0.8 * dataset_size))
    val_indices = indices[train_split:]
    test_dataset = Subset(db, val_indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
else:
    raise ValueError(f"Unsupported dataset: {args.dataset}")

print(f"Total dataset size: {dataset_size}")
if args.dataset == 'lidc' or args.dataset == 'isic':
    test_size = len(test_indices)
elif args.dataset == 'qubiq':
    test_size = dataset_size  # QUBIQ uses the entire provided dataset
elif args.dataset == 'isbi':
    test_size = len(val_indices)  # ISBI uses 20% for validation
else:
    test_size = dataset_size
print(f"Test set size: {test_size}")

# Hyperparameter: number of samples per network
samples_per_net = args.total_samples // len(networks)

# Evaluate the model
for i_batch, sampled_batch in enumerate(test_loader):
    print(f'Processing batch {i_batch}')
    logging.info(f'Processing batch {i_batch}')
    image_batch, label_batch = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
    label_four_batch = sampled_batch['label_four']
    image_batch_oc = sampled_batch['image_oc'].to(device)
    box1024_batch = sampled_batch['box_1024'].to(device)
    boxshift_batch = sampled_batch['box_shift'].to(device)
    pred_list = [[] for _ in range(image_batch.shape[0])]

    for net, weights in networks:
        for _ in range(samples_per_net):  # Generate multiple samples per network
            with torch.no_grad():
                outputs = net.forward(image_batch, image_batch_oc, box1024_batch, boxshift_batch, label_batch, device, train=False)
                logits_high = outputs['masks'].to(device) * weights.unsqueeze(-1)
                logits_high_res = logits_high.sum(1).unsqueeze(1)

            for j in range(image_batch.shape[0]):
                pred_list[j].append(logits_high_res[j])

    for index in range(len(pred_list)):
        pred_eval = torch.cat(pred_list[index], 0)
        pred_eval = (pred_eval > 0).cpu().detach().int()

        # Convert ground-truth to tensor [M,1,H,W]
        # 所有数据集（LIDC、QUBIQ、ISBI）现在都返回 numpy 数组格式
        label_four_arr = label_four_batch[index]
        if isinstance(label_four_arr, np.ndarray):
            if label_four_arr.ndim == 3:  # [M,H,W]
                gt_tensor = torch.from_numpy(label_four_arr).unsqueeze(1).int()
            elif label_four_arr.ndim == 4:  # [M,1,H,W]
                gt_tensor = torch.from_numpy(label_four_arr).int()
            else:  # [H,W] fallback
                gt_tensor = torch.from_numpy(label_four_arr).int().unsqueeze(0).unsqueeze(0)
        else:
            # 备用方案
            gt_tensor = torch.as_tensor(label_four_arr).int()
            if gt_tensor.ndim == 2:  # [H, W]
                gt_tensor = gt_tensor.unsqueeze(0).unsqueeze(0)
            elif gt_tensor.ndim == 3:  # [M, H, W]
                gt_tensor = gt_tensor.unsqueeze(1)

        ed_iter, _, (Dxy, Dxx, Dyy) = generalized_energy_distance_iou(pred_eval, gt_tensor)
        #_, ed_iter, (Dxy, Dxx, Dyy) = generalized_energy_distance_iou(pred_eval, gt_tensor)
        score = hm_iou_cal(pred_eval, gt_tensor)
        # use gt_tensor for dice_max as well to ensure shape [M,1,H,W]
        dice_max2_score += dice_max_cal2(pred_eval, gt_tensor)
        hm_iou_score += score
        ged_score += ed_iter
        Dxy_score += Dxy
        Dxx_score += Dxx
        Dyy_score += Dyy
        self_similarity_score += self_similarity_cal(pred_eval)
        bhattacharyya_distance_score += bhattacharyya_distance_cal(pred_eval)

# Calculate average scores
Dxy_avg = Dxy_score / test_size
Dxx_avg = Dxx_score / test_size
Dyy_avg = Dyy_score / test_size
ged = ged_score / test_size
dice_max2 = dice_max2_score / test_size
hm_iou = hm_iou_score / test_size
self_similarity = self_similarity_score / test_size
bhattacharyya_distance = bhattacharyya_distance_score / test_size

print(
    f"Dxy: {Dxy_avg}, Dxx: {Dxx_avg}, Dyy: {Dyy_avg}, ged_score: {ged}, "
    f"dice_max_score2: {dice_max2}, hm_iou_score: {hm_iou}, "
    f"self_similarity: {self_similarity}, bhattacharyya_distance: {bhattacharyya_distance}"
)
logging.info(
    f"Dxy: {Dxy_avg}, Dxx: {Dxx_avg}, Dyy: {Dyy_avg}, ged_score: {ged}, "
    f"dice_max_score2: {dice_max2}, hm_iou_score: {hm_iou}, "
    f"self_similarity: {self_similarity}, bhattacharyya_distance: {bhattacharyya_distance}"
)
