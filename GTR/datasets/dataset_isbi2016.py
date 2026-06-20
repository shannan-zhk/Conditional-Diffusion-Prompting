import os
import random
from typing import List

import glob2 as glob
import numpy as np
import PIL.Image as Image

import torch
import torchvision.transforms.functional as TF


class ISBI2016(torch.utils.data.Dataset):
    """
    ISBI 2016 ISIC Part 1 皮肤病变分割数据集

    目录结构（对应当前仓库中的 ISBI 文件夹）:
        data_path/
            ISBI2016_ISIC_Part1_Training_Data/
                ISIC_0000000.jpg
                ...
            ISBI2016_ISIC_Part1_Training_GroundTruth/
                ISIC_0000000_Segmentation.png
                ...

    注意：该数据集每张图像只有 1 个标注，我们为了兼容现有 GTR 代码中
    “多标注”接口，简单地将同一个 mask 复制 3 份返回。
    """

    def __init__(self, transform, apply_symmetric_transforms: bool, data_path: str):
        # 变换与是否做对称数据增强
        self.transform = transform
        self.symmetric_transforms = apply_symmetric_transforms

        self.images_dir = os.path.join(
            data_path, "ISBI2016_ISIC_Part1_Training_Data"
        )
        self.masks_dir = os.path.join(
            data_path, "ISBI2016_ISIC_Part1_Training_GroundTruth"
        )

        # 所有图像路径
        image_paths = sorted(glob.glob(os.path.join(self.images_dir, "*.jpg")))

        mask_paths: List[str] = []
        for img_path in image_paths:
            basename = os.path.basename(img_path)  # ISIC_0000000.jpg
            stem = os.path.splitext(basename)[0]  # ISIC_0000000
            mask_name = f"{stem}_Segmentation.png"
            mask_path = os.path.join(self.masks_dir, mask_name)
            if os.path.exists(mask_path):
                mask_paths.append(mask_path)
            else:
                # 若缺少掩膜，直接跳过该样本
                continue

        # 保证 image_paths 与 mask_paths 一一对应
        # 重新过滤只保留有 mask 的图像
        valid_image_paths: List[str] = []
        valid_mask_paths: List[str] = []
        for img_path in image_paths:
            basename = os.path.basename(img_path)
            stem = os.path.splitext(basename)[0]
            mask_name = f"{stem}_Segmentation.png"
            mask_path = os.path.join(self.masks_dir, mask_name)
            if os.path.exists(mask_path):
                valid_image_paths.append(img_path)
                valid_mask_paths.append(mask_path)

        self.image_paths = valid_image_paths
        self.mask_paths = valid_mask_paths

    def __len__(self):
        return len(self.image_paths)

    def symmetric_augmentation(self, images_and_masks: List[torch.Tensor]):
        """
        与 ISIC 数据集一致的对称几何增强：随机翻转 + 轻微旋转/平移/缩放/剪切
        """
        # 水平翻转
        if np.random.random() > 0.5:
            images_and_masks = [TF.hflip(x) for x in images_and_masks]

        # 垂直翻转
        if np.random.random() > 0.5:
            images_and_masks = [TF.vflip(x) for x in images_and_masks]

        # 随机仿射变换
        angle = random.randint(-15, 15)
        translation = (random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05))
        scale = random.uniform(0.9, 1.1)
        shear = (random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))

        images_and_masks = [
            TF.affine(
                x, angle=angle, translate=translation, scale=scale, shear=shear, fill=0
            )
            for x in images_and_masks
        ]
        return images_and_masks

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        def path_to_image(path):
            img = Image.open(path)
            return img

        def to_tensor(x):
            x = path_to_image(x)
            x = self.transform(x)
            x = x.float()
            return x

        # 准备为几何增强的 PIL 图像
        img_pil = path_to_image(image_path)
        mask_pil = path_to_image(mask_path)

        pil_list = [img_pil, mask_pil]
        if self.symmetric_transforms:
            pil_list = self.symmetric_augmentation(pil_list)
        img_pil, mask_pil = pil_list

        # 再做 ToTensor 等变换
        image = self.transform(img_pil).float()
        mask = self.transform(mask_pil).float()

        # 兼容原始接口：返回 1 个 target + 3 个“标注者”掩膜 + style 标签
        target = mask
        target1 = mask
        target2 = mask
        target3 = mask
        # ISBI2016 没有区分风格，这里统一设为 0，类型为张量以避免 DataLoader collate 报错
        style = torch.as_tensor(0, dtype=torch.long)

        return image, target, [target1, target2, target3], style


