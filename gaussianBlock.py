"""
PyTorch implementation of the GaussianDiffusion block
originally defined in LDSeg-main/utilities/gaussianBlock.py.

仅实现前向扩散 q(x_t | x_0)：
    x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon
"""

import numpy as np
import torch


def cosine_func(t, T, s: float = 0.008):
    """
    Cosine schedule 函数，对应 LDSeg-main 中的 cosineFunc。
    t: numpy array of timesteps
    T: total timesteps
    """
    return np.cos(((t / T + s) / (1.0 + s)) * (np.pi / 2.0)) ** 2


class GaussianDiffusion:
    """Gaussian block (PyTorch version)."""

    def __init__(
        self,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        timesteps: int = 1000,
        schedule: str = "cosine",
    ):
        # 保存参数
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps

        # 根据不同 schedule 生成 betas / alphas / alpha_cumprod
        if schedule == "linear":
            betas = np.linspace(
                beta_start,
                beta_end,
                timesteps,
                dtype=np.float64,
            )
            times = np.arange(0, timesteps)
            alphas = 1.0 - betas
            alphas_cumprod = np.cumprod(alphas, axis=0)
            alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        elif schedule == "cosine":
            times = np.arange(0, timesteps)
            alphas_cumprod = cosine_func(times, timesteps) / cosine_func(
                np.zeros_like(times), timesteps
            )
            alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
            alphas = alphas_cumprod / alphas_cumprod_prev
            betas = 1.0 - alphas

        elif schedule == "quadratic":
            order = 2
            times = np.arange(0, timesteps)
            betas = np.power(
                np.linspace(
                    beta_end ** (1.0 / order),
                    beta_start ** (1.0 / order),
                    timesteps,
                ),
                order,
            )[::-1]
            alphas = 1.0 - betas
            alphas_cumprod = np.cumprod(alphas, axis=0)
            alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # 存成 PyTorch tensor（默认在 CPU，上层使用时会 to(device)）
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32)
        self.alphas_cumprod_prev = torch.tensor(alphas_cumprod_prev, dtype=torch.float32)
        self.sqrt_alphas_cumprod = torch.tensor(
            np.sqrt(alphas_cumprod), dtype=torch.float32
        )
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(
            np.sqrt(1.0 - alphas_cumprod), dtype=torch.float32
        )

    def _extract(self, a: torch.Tensor, t: torch.LongTensor, x_shape):
        """
        从一维数组 a 中按时间步 t 取出对应系数，并 reshape 成 [B,1,1,1] 方便与特征相乘。
        a: [T]
        t: [B] long
        x_shape: 目标张量形状，用来取 batch_size
        """
        batch_size = x_shape[0]
        # 保证 a 和 t 在同一个 device 上
        a = a.to(t.device)
        out = a.gather(0, t)  # [B]
        return out.view(batch_size, 1, 1, 1)

    def q_sample(self, x_start: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor):
        """
        前向扩散：给定 x_start、时间步 t 和噪声 epsilon，生成 x_t。
        x_start: [B, C, H, W]
        t: [B] long，取值范围 [0, timesteps)
        noise: [B, C, H, W]
        """
        assert x_start.shape == noise.shape
        sqrt_alpha_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alpha_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.to(x_start.device)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.to(
            x_start.device
        )
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise


