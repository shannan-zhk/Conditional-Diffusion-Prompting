# Conditional Diffusion Prompting for Ambiguous Medical Image Segmentation

<div align="center">

**MICCAI 2026**

[![MICCAI](https://img.shields.io/badge/MICCAI-2026-blue)]()
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-red)]()

**HongKai Zhao, Jun Gao, Qingbo Kang, Qicheng Lao**

Beijing University of Posts and Telecommunications · Stork Healthcare · West China Hospital, Sichuan University

</div>

---

## News

- **2026-06**: Our paper "Conditional Diffusion Prompting for Ambiguous Medical Image Segmentation" has been accepted at **MICCAI 2026**! 🎉

## Abstract

Ambiguous medical image segmentation often admits multiple clinically plausible delineations due to substantial inter-expert variability, and such inherent uncertainty cannot be faithfully represented by a single deterministic mask. Promptable foundation models such as SAM offer strong representations, yet under ambiguous boundaries, their predictions can be highly sensitive to prompt perturbations, and existing stochastic prompting strategies are often coarse and weakly coupled to image-side ambiguity, yielding diverse samples that may deviate from image-consistent boundaries.

We propose **Conditional Diffusion Prompting (CDP)**, which performs diffusion prompting in dense prompt embedding space prior to SAM-style decoding and conditions prompt-space denoising on stochastic image embeddings to couple prompt ambiguity with image ambiguity. Experiments on multiple ambiguous medical image segmentation benchmarks demonstrate improved distribution matching and sample quality, achieving lower GED and higher HM-IOU and D<sub>max</sub>. CDP provides a structured sampling mechanism for promptable decoders, enabling diverse yet image-consistent segmentations and improving practical reliability in ambiguous clinical settings.

---

## Architecture

```
├── asam.py                        # ASAM model (SAM-based ambiguous segmentation)
├── vaeasam.py                     # VAE-ASAM variant with variational inference
├── sam_lora_image_encoder.py      # LoRA fine-tuned SAM image encoder
├── gaussianBlock.py               # Gaussian diffusion block
├── utils.py                       # Utility functions
├── train_lidc.py                  # Training script for LIDC dataset
├── evaluate_lidc.py               # Evaluation script for LIDC dataset
├── requirements.txt               # Python dependencies
├── GTR/
│   ├── models/
│   │   ├── GTR.py                 # Gaussian Truncation Representation
│   │   ├── Condition_Unet.py      # Conditional U-Net for diffusion
│   │   ├── unet.py                # Base U-Net
│   │   └── unet-mnist.py          # U-Net for MNIST
│   ├── datasets/                  # Dataset loaders (ISIC, LIDC, ISBI, etc.)
│   ├── dataloaders.py             # Data loading utilities
│   ├── metadata_manager.py        # Metadata management
│   └── utils/
│       ├── loss.py                # Loss functions
│       ├── metrics.py             # Evaluation metrics
│       └── utils.py               # General utilities
├── segment_anything/              # SAM dependency (Meta SAM)
└── README.md
```

---

## Installation

```bash
git clone https://github.com/shannan-zhk/Conditional-Diffusion-Prompting.git
cd Conditional-Diffusion-Prompting
pip install -r requirements.txt
```

### Pretrained SAM Weights

Download the SAM ViT-B weights and place them in the project root:

```bash
# Option 1: Download from Meta SAM
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Option 2: Already included in this repo
```

---

## Usage

### Training

```bash
python train_lidc.py --batch-size 16 --epochs 100 --lr 1e-4
```

### Evaluation

```bash
python evaluate_lidc.py --checkpoint path/to/checkpoint.pth
```

---

## Datasets

This project supports multiple medical image segmentation datasets with inter-annotator variability:

| Dataset | Modality | Target | Annotators |
|---------|----------|--------|------------|
| **LIDC** | CT | Lung nodules | 4 radiologists |
| **ISIC** | Dermoscopy | Skin lesions | Multiple experts |
| **ISBI** | Various | Challenge datasets | Multiple |
| **QUBIQ** | Various (Prostate, Pancreatic-lesion) | Uncertainty quantification | Multiple |

---

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{zhao2026cdp,
    title={Conditional Diffusion Prompting for Ambiguous Medical Image Segmentation},
    author={Zhao, HongKai and Gao, Jun and Kang, Qingbo and Lao, Qicheng},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
    year={2026}
}
```

---

## License

This project is licensed under the MIT License.

## Contact

- **HongKai Zhao** — zhk2336258035@bupt.edu.cn
- **Qicheng Lao** — qicheng.lao@gmail.com

GitHub: [@shannan-zhk](https://github.com/shannan-zhk)
