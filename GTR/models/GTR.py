import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributions as td

# 同时兼容两种使用方式：
# 1) 在 GTR 子项目中直接运行 `python train_GTR.py`（当前目录为 GTR/）：
#       需要 `from utils.utils` / `from models.unet` 这种绝对导入方式；
# 2) 在上层工程中通过 `from GTR.models.GTR import GaussianTruncationRepresentation` 导入：
#       需要包内相对导入 `from ..utils.utils` / `from .unet`。
try:
    # 情况 2：作为 GTR 包的一部分被上层工程导入
    from ..utils.utils import *  # type: ignore
    from .unet import Unet  # type: ignore
except Exception:
    try:
        # 情况 1：在 GTR 目录下直接运行 train_GTR.py，但通过包名绝对导入，避免被上层 utils.py 遮蔽
        from GTR.utils.utils import *  # type: ignore
        from GTR.models.unet import Unet  # type: ignore
    except Exception:
        # 兜底：仍尝试相对当前工作目录的简易导入
        from utils.utils import *  # type: ignore
        from models.unet import Unet  # type: ignore


class GaussianTruncationRepresentation(nn.Module):
    def __init__(
        self,
        name,
        num_channels=1,
        num_classes=1,
        num_filters=[32, 64, 128, 192],
        rank: int = 10,
        epsilon=1e-5,
        diagonal=False,
    ):
        super().__init__()
        self.name = name
        self.rank = rank
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.epsilon = epsilon
        conv_fn = nn.Conv2d
        # whether to use only the diagonal (independent normals)
        self.diagonal = diagonal
        self.mean_l = conv_fn(num_filters[0], num_classes, kernel_size=1)
        self.log_cov_diag_l = conv_fn(num_filters[0], num_classes, kernel_size=1)
        self.cov_factor_l = conv_fn(num_filters[0], num_classes * rank, kernel_size=1)

        self.unet = Unet(
            name=self.name,
            input_channels=self.num_channels,
            num_classes=self.num_classes,
            num_filters=[32, 64, 128, 192],
            apply_last_layer=False,
        )

    def forward(self, image):
        # 前向传播方法：输入图像，输出高斯截断表示的预测结果
        logits = self.unet.forward(image)  # 通过UNet网络处理输入图像，获得特征logits
        batch_size = logits.shape[0]  # 获取批次大小，即当前批次的样本数量

        # 定义事件形状：(类别数, 高度, 宽度)，用于概率分布的输出维度
        # tensor size num_classesxHxW
        event_shape = (self.num_classes,) + logits.shape[2:]  # 从logits的最后两个维度获取H和W

        # 计算多元高斯分布的参数
        mean = self.mean_l(logits)  # 使用1x1卷积计算每个像素的均值参数
        # 使用 softplus 而非 exp 来参数化协方差对角，避免方差过大/过小导致数值不稳定
        cov_diag = F.softplus(self.log_cov_diag_l(logits)) + self.epsilon
        # 将每个样本展平，便于后续概率分布计算
        # Flattens out each image in the batch, size is batchsize x (rest)
        mean = mean.view((batch_size, -1))  # 重塑均值张量：(batch_size, num_classes*H*W)
        cov_diag = cov_diag.view((batch_size, -1))  # 重塑对角协方差张量：(batch_size, num_classes*H*W)

        # 计算协方差因子的低秩近似参数
        cov_factor = self.cov_factor_l(logits)  # 使用1x1卷积计算协方差因子
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))  # 重塑为：(batch_size, rank, num_classes, H*W)
        cov_factor = cov_factor.flatten(2, 3)  # 展平类别和空间维度：(batch_size, rank, num_classes*H*W)
        cov_factor = cov_factor.transpose(1, 2)  # 转置得到：(batch_size, num_classes*H*W, rank)

        # 冗余赋值，保持代码一致性
        cov_factor = cov_factor
        # 适度下界，进一步避免极小方差导致协方差矩阵病态
        cov_diag = cov_diag.clamp_min(self.epsilon)

        # 创建用于训练循环日志记录的信息字典
        # A dictionary that is handed over to the training loop for logging
        infos_for_logging = {
            "mean": mean,  # 均值参数
            "cov_factor": cov_factor,  # 协方差因子
            "cov_diag": cov_diag,  # 对角协方差
            "Max value of mean": torch.max(mean),  # 均值的最大值，用于监控训练稳定性
            "Min value of mean": torch.min(mean),  # 均值的最小值
            "Max Value of Cov_diag": torch.max(cov_diag),  # 对角协方差的最大值
            "Max Value of Cov_factor": torch.max(cov_factor),  # 协方差因子的最大值
        }

        # 根据配置选择分布类型
        if self.diagonal:  # 如果使用对角协方差（独立高斯分布）
            base_distribution = td.Independent(  # 创建独立的正态分布
                td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1  # 每个维度独立，scale为标准差
            )
        else:  # 使用低秩协方差近似
            try:
                base_distribution = td.LowRankMultivariateNormal(  # 创建低秩多元正态分布
                    loc=mean, cov_factor=cov_factor, cov_diag=cov_diag  # 使用均值、低秩因子和对角协方差
                )
            except:  # 如果协方差矩阵不可逆（数值不稳定）
                print(  # 打印警告信息
                    "Covariance became not invertible. Using independent normals for this batch!"
                )
                base_distribution = td.Independent(  # 回退到独立正态分布
                    td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1
                )

        # 创建重塑分布，将基础分布重塑为所需的事件形状
        distribution = ReshapedDistribution(
            base_distribution=base_distribution,  # 基础概率分布
            new_event_shape=event_shape,  # 新的事件形状：(num_classes, H, W)
            validate_args=False,  # 禁用参数验证以提高性能
        )

        # 重塑输出张量为原始图像维度
        shape = (batch_size,) + event_shape  # 完整输出形状：(batch_size, num_classes, H, W)
        logit_mean = mean.view(shape)  # 将展平的均值重塑为原始形状
        cov_diag_view = cov_diag.view(shape).detach()  # 重塑对角协方差并分离梯度
        cov_factor_view = (  # 重塑协方差因子
            cov_factor.transpose(2, 1)  # 转置回：(batch_size, rank, num_classes*H*W)
            .view((batch_size, self.num_classes * self.rank) + event_shape[1:])  # 重塑为：(batch_size, num_classes*rank, H, W)
            .detach()  # 分离梯度
        )

        # 创建输出字典，包含所有预测结果
        output_dict = {
            "logit_mean": logit_mean.detach(),  # 分离梯度的logit均值
            "cov_diag": cov_diag_view,  # 对角协方差视图
            "cov_factor": cov_factor_view,  # 协方差因子视图
            "distribution": distribution,  # 完整的概率分布对象
        }

        # 返回预测的logit均值、输出字典和日志信息
        return logit_mean, output_dict, infos_for_logging