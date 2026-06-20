import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from lidc_dataset import LIDC_IDRI, RandomGenerator
from utils import (
    l2_regularisation,
    kl_divergence,
    calculate_dice_loss,
    calculate_sigmoid_focal_loss,
)
from torchvision import transforms
from tensorboardX import SummaryWriter
from asam import ASAM
import argparse
import random

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Training settings')
parser.add_argument('--device', type=str, default='cuda:3', help='Device to use for training')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
parser.add_argument('--max_epoch', type=int, default=50, help='Maximum number of epochs')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer')
parser.add_argument(
    '--lambda_diff',
    type=float,
    default=0.05,
    help='扩散分支 diff_loss 的损失权重（默认 0.05，可按稳定性/显存调整）'
)
parser.add_argument('--save_path', type=str, default='/home/u2024111264/share/zhk/asam原代码/checkpoint', help='Path to save combined weights')
parser.add_argument('--dataset', type=str, default='lidc', choices=['lidc', 'qubiq', 'isbi', 'isic'], help='Choose dataset: lidc, qubiq, isbi, or isic')
parser.add_argument('--qubiq_train_dir', type=str, default='QUBIQ/training_data_v3_QC', help='QUBIQ training dir root')
parser.add_argument('--qubiq_val_dir', type=str, default='QUBIQ/validation_data_qubiq2021_QC', help='QUBIQ validation dir root')
parser.add_argument('--qubiq_task', type=str, default=None, help='QUBIQ task name, e.g., prostate, brain-growth, brain-tumor, pancreas, pancreatic-lesion')
parser.add_argument('--isbi_image_dir', type=str, default='/home/u2024111264/share/zhk/asam原代码/ISBI/ISBI2016_ISIC_Part1_Training_Data', help='ISBI images directory')
parser.add_argument('--isbi_mask_dir', type=str, default='/home/u2024111264/share/zhk/asam原代码/ISBI/ISBI2016_ISIC_Part1_Training_GroundTruth', help='ISBI masks directory')
parser.add_argument('--isic_dir', type=str, default='/home/u2024111264/share/zhk/GTRasam/GTR/data/isic256_3_style', help='ISIC dataset root directory (Images/ and Segmentations/ subfolders)')
parser.add_argument('--posterior_fixed_std_normal', action='store_true', help='Fix posterior as standard Normal N(0, I) during training')
args = parser.parse_args()

# Ensure save directory exists
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

# Set device
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')


# Define MaskWeights class
class MaskWeights(nn.Module):
    def __init__(self):
        super(MaskWeights, self).__init__()
        self.weights = nn.Parameter(torch.ones(5, 1, requires_grad=True) / 6)

# Initialize network and mask weights
# 注意：将当前选择的数据集名称传入 ASAM，用于决定 GTR 的输入通道数等行为
net = ASAM(
    posterior_fixed_std_normal=args.posterior_fixed_std_normal,
    gtr_ckpt_path="/home/u2024111264/share/zhk/GTRasamdiffusion/GTR/saved_models/LIDC/2025_12_05_02_18/best_model.pt",
    dataset=args.dataset,
).to(device)

mask_weights = MaskWeights().to(device)
mask_weights.train()

# Load dataset
if args.dataset == 'lidc':
    db = LIDC_IDRI(dataset_location='LIDC/data/', transform=transforms.Compose([
        RandomGenerator(output_size=[128, 128])
    ]))
elif args.dataset == 'qubiq':
    from qubiq_dataset import QUBIQ_Dataset
    train_base = args.qubiq_train_dir if args.qubiq_task is None else os.path.join(args.qubiq_train_dir, args.qubiq_task)
    val_base = args.qubiq_val_dir if args.qubiq_task is None else os.path.join(args.qubiq_val_dir, args.qubiq_task)
    db_train = QUBIQ_Dataset(
        base_dir=train_base,
        target_size=(128, 128),
        raters_per_sample=None,
        task_name=args.qubiq_task,
        transform=transforms.Compose([RandomGenerator(output_size=[128, 128])])
    )
    db_val = QUBIQ_Dataset(
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
    # 使用我们为 ASAM 自定义的 ISIC 数据集加载器
    from isic_dataset import ISIC_Dataset
    db = ISIC_Dataset(
        base_dir=args.isic_dir,
        target_size=(128, 128),
    )
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")
if args.dataset == 'lidc':
    dataset_size = len(db)
    indices = list(range(dataset_size))
    train_split = int(np.floor(0.6 * dataset_size))
    validation_split = int(np.floor(0.8 * dataset_size))

    train_indices = indices[:train_split]
    validation_indices = indices[train_split:validation_split]
    test_indices = indices[validation_split:]

    train_dataset = Subset(db, train_indices)
    validation_dataset = Subset(db, validation_indices)
    test_dataset = Subset(db, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Total dataset size: {dataset_size}")
    print(f"Training set size: {len(train_indices)}")
    print(f"Validation set size: {len(validation_indices)}")
    print(f"Test set size: {len(test_indices)}")
elif args.dataset == 'qubiq':
    print(f"QUBIQ train samples: {len(db_train)}")
    print(f"QUBIQ validation samples: {len(db_val)}")
    train_loader = DataLoader(db_train, batch_size=args.batch_size, shuffle=False)
    validation_loader = DataLoader(db_val, batch_size=1, shuffle=False)
    test_loader = validation_loader
elif args.dataset == 'isbi':
    dataset_size = len(db)
    indices = list(range(dataset_size))
    train_split = int(np.floor(0.8 * dataset_size))
    train_indices = indices[:train_split]
    val_indices = indices[train_split:]
    train_dataset = Subset(db, train_indices)
    validation_dataset = Subset(db, val_indices)
    test_dataset = validation_dataset

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    test_loader = validation_loader
elif args.dataset == 'isic':
    # 仿照 LIDC，将 ISIC 划分为 60% 训练, 20% 验证, 20% 测试
    dataset_size = len(db)
    indices = list(range(dataset_size))
    train_split = int(np.floor(0.6 * dataset_size))
    validation_split = int(np.floor(0.8 * dataset_size))

    train_indices = indices[:train_split]
    validation_indices = indices[train_split:validation_split]
    test_indices = indices[validation_split:]

    train_dataset = Subset(db, train_indices)
    validation_dataset = Subset(db, validation_indices)
    test_dataset = Subset(db, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Total dataset size (ISIC): {dataset_size}")
    print(f"Training set size: {len(train_indices)}")
    print(f"Validation set size: {len(validation_indices)}")
    print(f"Test set size: {len(test_indices)}")

# Initialize TensorBoard writer and optimizer
writer = SummaryWriter('tf-logs/train_onestage')
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

combined_weights = {}

# Training loop
for epoch_num in range(1, args.max_epoch + 1):
    net.train()
    loss_epoch = 0.0
    segloss_epoch = 0.0
    diffloss_epoch = 0.0
    if args.dataset == 'qubiq' or args.dataset == 'isic':
        if args.qubiq_task in ['pancreas', 'pancreatic-lesion']:
            print(f"Epoch {epoch_num}")
        else:
            if epoch_num % 50 == 0:
                print(f"Epoch {epoch_num}")
    else:
        print(f"Epoch {epoch_num}")

    for i_batch, sampled_batch in enumerate(train_loader):
        image_batch, label_batch = sampled_batch['image'].to(device), sampled_batch['label'].to(device) #image_batch是输入图像将单通道图像 repeat 成3通道，label_batch是标签图像随机从4个标注者中抽取1个掩码作为当前样本的 label
        image_batch_oc = sampled_batch['image_oc'].to(device)  #image_batch_oc是原始图像副本，用于后续的对比学习
        box1024_batch = sampled_batch['box_1024'].to(device)  #box1024_batch是1024x1024的框，用于后续的对比学习， 随机抽到的该标注者掩码非空，用该掩码的 bbox
        boxshift_batch = sampled_batch['box_shift'].to(device) #以“最宽最长”bbox 为基准，随机缩放/平移后的框

        outputs = net(image_batch, image_batch_oc, box1024_batch, boxshift_batch, label_batch, device)
        output_masks = outputs['masks']
        logits_high = output_masks.to(device)
        weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0).to(device)
        logits_high = logits_high * weights.unsqueeze(-1)
        logits_high_res = logits_high.sum(1).unsqueeze(1)

        #kl1 = torch.mean(kl_divergence(net.posterior_box_latent_space, net.prior_box_latent_space))
        kl2 = torch.mean(kl_divergence(net.posterior_object_latent_space, net.prior_object_latent_space))
        cel_loss = nn.CrossEntropyLoss()(logits_high, label_batch.long())
        # reg_loss = sum(l2_regularisation(layer) for layer in [
        #     net.prior_box, net.posterior_box, net.fcomb_box.layers,
        #     net.prior_object, net.posterior_object, net.fcomb_object.layers
        # ])
        reg_loss = sum(l2_regularisation(layer) for layer in [
            net.prior_object, net.posterior_object, net.fcomb_object.layers
        ])
        gt_mask = label_batch.unsqueeze(1)
        dice_loss = calculate_dice_loss(logits_high_res, gt_mask.long())
        focal_loss = calculate_sigmoid_focal_loss(logits_high_res, gt_mask.float())
        seg_loss = cel_loss + dice_loss + focal_loss
        #seg_loss = cel_loss + dice_loss     #删掉focal_loss
        #loss = seg_loss + 1e-5 * reg_loss + kl2

        # 扩散分支损失：若模型未返回 diff_loss，则视为 0
        diff_loss = outputs.get('diff_loss', None)
        diff_loss_term = diff_loss if diff_loss is not None else torch.tensor(0.0, device=device)

        #loss = seg_loss + 1e-5 * reg_loss + kl1 + kl2 + args.lambda_diff * diff_loss_term
        loss = seg_loss + 1e-5 * reg_loss  + 0.8*(kl2 + args.lambda_diff * diff_loss_term)


        segloss_epoch += seg_loss.item()
        loss_epoch += loss.item()
        diffloss_epoch += float(diff_loss_term.detach().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 仅打印训练损失（保持与添加验证前一致的行为）
    if args.dataset == 'qubiq' or args.dataset == 'isic':
        if args.qubiq_task in ['pancreas', 'pancreatic-lesion']:
            print(f"Segmentation loss: {segloss_epoch / len(train_loader)}")
            print(f"Diffusion loss: {diffloss_epoch / len(train_loader)} (λ={args.lambda_diff})")
            print(f"Total loss: {loss_epoch / len(train_loader)}")
        else:
            if epoch_num % 50 == 0:
                print(f"Segmentation loss: {segloss_epoch / len(train_loader)}")
                print(f"Diffusion loss: {diffloss_epoch / len(train_loader)} (λ={args.lambda_diff})")
                print(f"Total loss: {loss_epoch / len(train_loader)}")
    else:
        print(f"Segmentation loss: {segloss_epoch / len(train_loader)}")
        print(f"Diffusion loss: {diffloss_epoch / len(train_loader)} (λ={args.lambda_diff})")
        print(f"Total loss: {loss_epoch / len(train_loader)}")

    # Save model weights frequency
    if args.dataset == 'qubiq' or args.dataset == 'isic':
        save_every = 10 if args.qubiq_task in ['pancreas', 'pancreatic-lesion'] else 50
    else:
        save_every = 10
    if epoch_num % save_every == 0:
        combined_weights[f'epoch_{epoch_num}'] = {
            'model_state_dict': {k: v.cpu() for k, v in net.state_dict().items()},
            'mask_weights': weights.cpu()
        }
        print(f"Saved weights for epoch {epoch_num}")

# Save all weights to file
torch.save(combined_weights, args.save_path)
print(f"Combined weights saved to {args.save_path}.")
