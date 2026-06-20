import math
import numpy as np
import torch.nn as nn
import torch
from segment_anything import sam_model_registry
from sam_lora_image_encoder import LoRA_Sam
from utils import init_weights,init_weights_orthogonal_normal
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from gaussianBlock import GaussianDiffusion

# 可选：从 GTR 子项目中引入先验网络，用作 ASAM 的“冻结 GTR 分支”
# 当前工程根目录下已有 `GTR/` 包，内部相对导入使用 `from ..utils.utils` 等，
# 所以这里直接按包名 `GTR.models.GTR` 导入即可。
try:
    from GTR.models.GTR import GaussianTruncationRepresentation
except Exception as e:
    GaussianTruncationRepresentation = None
    print(f"[ASAM] Warning: failed to import GTR GaussianTruncationRepresentation: {e}")


class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers,padding=True,posterior=False,object=False,dataset='lidc'):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.posterior=posterior
        self.object=object
        if dataset=='lidc':
            if self.posterior and self.object:
                self.input_channels += 1
        else:
            if self.posterior and self.object:
                self.input_channels += 3
            elif self.posterior==False and self.object:
                self.input_channels += 2

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output



class AxisAlignedConvGaussian_box(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False,object=False,dataset='lidc'):
        super(AxisAlignedConvGaussian_box, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.output_channels=8
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        self.object=object
        self.dataset=dataset
        if self.posterior:
            self.name = 'Posterior'
            self.input_feature=1024
        else:
            self.name = 'Prior'
            self.input_feature=512
        self.box_input_channel=264
        self.fc1 = nn.Linear(self.input_feature, 1024) 
        self.fc2 = nn.Linear(1024, 512)             
        self.fc3 = nn.Linear(512, self.output_channels * 64)  
        self.box_conv = nn.Sequential(
            nn.Conv2d(self.box_input_channel, 128, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 8x8 -> 16x16
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 16x16 -> 16x16
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 16x16 -> 32x32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32x32 -> 32x32
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 32x32 -> 64x64
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 64x64 -> 64x64
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 64x64 -> 128x128
            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # 128x128 -> 128x128
        )


        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers,posterior=self.posterior, object=self.object,dataset=self.dataset)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input,boxemb_shift,boxemb_ori=None):
        if boxemb_ori is not None:
            boxemb_input = torch.cat((boxemb_shift, boxemb_ori), dim=1)
        else:
            boxemb_input = boxemb_shift
        
        boxemb_input = boxemb_input.view(boxemb_input.size(0), -1)
        boxemb_input = F.relu(self.fc1(boxemb_input))
        boxemb_input = F.relu(self.fc2(boxemb_input))
        boxemb_input = self.fc3(boxemb_input)

        boxemb_input = boxemb_input.view(boxemb_input.size(0), self.output_channels, 8, 8)
        input=torch.cat((input, boxemb_input), dim=1)
        input = input.to(torch.float32)
        input=self.box_conv(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        #We only want the mean of the resulting hxw image，
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        mu_log_sigma = self.conv_layer(encoding)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        
        return dist


class Fcomb_box(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb_box, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [1,2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb 
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(512, 256, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(256, 256, kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(256, 256, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
        return torch.index_select(a, dim, order_index)


    def forward(self, feature_map, z):
        if self.use_tile:
            # print(feature_map.shape)#torch.Size([1, 256, 8, 8])
            # print(z.shape)#torch.Size([1, 6])

            z = torch.unsqueeze(z,2)
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[1]])
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[2]])
            feature_map = torch.cat((feature_map, z), dim=1)
            output = self.layers(feature_map)
            output = self.last_layer(output)
            return output     




class AxisAlignedConvGaussian_object(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False,object=False,dataset='lidc'):
        super(AxisAlignedConvGaussian_object, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        self.object=object
        self.dataset=dataset
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers,posterior=self.posterior, object=self.object,dataset=self.dataset)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input,segm=None):
        if segm is not None:
            input = torch.cat((input, segm), dim=1)

        input = input.to(torch.float32)
        encoding = self.encoder(input)
        self.show_enc = encoding

        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)


        mu_log_sigma = self.conv_layer(encoding)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        
        return dist


class Fcomb_object(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb_object, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [1,2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb 
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(512, 256, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(256, 256, kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(256, 256, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:

            z = torch.unsqueeze(z,2)#
            z = torch.unsqueeze(z,2)#
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[1]])
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[2]])
            feature_map = torch.cat((feature_map, z), dim=1)
            output = self.layers(feature_map)
            output = self.last_layer(output)

            return output     

# ===== 扩散所需的时间嵌入与去噪器 =====

class TimeEmbeddingTorch(nn.Module):
    """
    时间步嵌入，参考 LDSeg 的 TimeEmbedding：正弦余弦位置编码。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        inv_freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t[None]
        t = t.float().view(-1)  # [B]
        sinusoid = t[:, None] * self.inv_freq[None, :]
        emb = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)
        return emb


class TimeMLPTorch(nn.Module):
    """时间嵌入的 MLP 头。"""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        return self.mlp(temb)


class ResidualBlockTorch(nn.Module):
    """
    2D 残差块（GroupNorm + SiLU + Conv），并注入时间嵌入。
    """
    def __init__(self, in_channels: int, out_channels: int, temb_dim: int, groups: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.temb_proj = nn.Linear(temb_dim, out_channels)

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        temb_proj = self.temb_proj(temb)  # [B, out_channels]
        h = h + temb_proj[:, :, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class PromptDenoiser(nn.Module):
    """
    扩散去噪器：输入加噪的 dense prompt 与图像特征，预测噪声。
    """
    def __init__(
        self,
        dense_channels: int,
        image_channels: int,
        hidden_channels: int = 128,
        num_res_blocks: int = 3,
        temb_dim: int = 256,
        groups: int = 8,
    ):
        super().__init__()
        in_channels = dense_channels + image_channels
        self.conv_in = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)

        self.time_embed = TimeEmbeddingTorch(temb_dim)
        self.time_mlp = TimeMLPTorch(temb_dim, temb_dim)

        self.resblocks = nn.ModuleList(
            [
                ResidualBlockTorch(
                    hidden_channels,
                    hidden_channels,
                    temb_dim=temb_dim,
                    groups=groups,
                )
                for _ in range(num_res_blocks)
            ]
        )

        self.norm_out = nn.GroupNorm(groups, hidden_channels)
        self.act = nn.SiLU()
        self.conv_out = nn.Conv2d(hidden_channels, dense_channels, kernel_size=3, padding=1)

    def forward(
        self,
        dense_t: torch.Tensor,
        image_embeddings: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([dense_t, image_embeddings], dim=1)
        x = self.conv_in(x)

        temb = self.time_embed(t)
        temb = self.time_mlp(temb)

        for block in self.resblocks:
            x = block(x, temb)

        x = self.norm_out(x)
        x = self.act(x)
        eps_hat = self.conv_out(x)
        return eps_hat

class ASAM(nn.Module):

    def __init__(
        self,
        input_channels=1,
        num_classes=6,
        img_size=128,
        num_filters=[32, 64, 128, 192],
        latent_dim=256,
        no_convs_fcomb=4,
        beta=10.0,
        dataset="lidc",
        posterior_fixed_std_normal: bool = False,
        gtr_ckpt_path: str = None,
    ):
        super(ASAM, self).__init__()
        self.ckpt="sam_vit_b_01ec64.pth"
        self.img_size=img_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {"w": "he_normal", "b": "normal"}
        self.beta = beta
        self.z_prior_sample = 0
        self.dataset = dataset
        # 若为 True，则在训练阶段将后验固定为标准正态 N(0, I)
        self.posterior_fixed_std_normal = bool(posterior_fixed_std_normal)
        self.sam, self.img_embedding_size = sam_model_registry["vit_b"](
            image_size=self.img_size,
            num_classes=self.num_classes,
            checkpoint=self.ckpt,
            pixel_mean=[0, 0, 0],
            pixel_std=[1, 1, 1],
        )
        self.lora_sam = LoRA_Sam(self.sam, 4)
        # 图像分支：始终按 LIDC 配置处理输入通道（1 通道原图 + 可选 1 通道 seg），
        # 这样无论当前数据集为何种模态，prior/posterior_object 的卷积期望与传入的
        # batch_input_ori / batch_mask 形状都保持一致，避免通道数不匹配。
        self.prior_object = AxisAlignedConvGaussian_object(
            self.input_channels,
            self.num_filters,
            self.no_convs_per_block,
            self.latent_dim,
            self.initializers,
            posterior=False,
            object=True,
            dataset="lidc",
        )
        self.prior_box = AxisAlignedConvGaussian_box(
            self.input_channels,
            self.num_filters,
            self.no_convs_per_block,
            self.latent_dim,
            self.initializers,
            posterior=False,
            object=False,
            dataset=self.dataset,
        )
        self.posterior_object = AxisAlignedConvGaussian_object(
            self.input_channels,
            self.num_filters,
            self.no_convs_per_block,
            self.latent_dim,
            self.initializers,
            posterior=True,
            object=True,
            dataset="lidc",
        )
        self.posterior_box = AxisAlignedConvGaussian_box(
            self.input_channels,
            self.num_filters,
            self.no_convs_per_block,
            self.latent_dim,
            self.initializers,
            posterior=True,
            object=False,
            dataset=self.dataset,
        )
        self.fcomb_object = Fcomb_object(
            self.num_filters,
            self.latent_dim,
            self.input_channels,
            self.num_classes,
            self.no_convs_fcomb,
            {"w": "orthogonal", "b": "normal"},
            use_tile=True,
        )
        self.fcomb_box = Fcomb_box(
            self.num_filters,
            self.latent_dim,
            self.input_channels,
            self.num_classes,
            self.no_convs_fcomb,
            {"w": "orthogonal", "b": "normal"},
            use_tile=True,
        )

        # prompt latent 的高斯扩散模块（前向 q_sample）
        self.prompt_diffusion = GaussianDiffusion(
            beta_start=1e-4,
            beta_end=0.02,
            timesteps=1000,
            schedule='cosine',
        )
        # prompt denoiser：SAM ViT-B 的 dense embedding 通道为 256，图像特征同样为 256
        dense_ch = 256
        img_ch = 256
        self.prompt_denoiser = PromptDenoiser(
            dense_channels=dense_ch,
            image_channels=img_ch,
            hidden_channels=128,
            num_res_blocks=3,
            temb_dim=256,
            groups=8,
        )
        # # 残差校正：扩散只提供增量，不覆盖 z0
        # self.residual_conv = nn.Sequential(
        #     nn.Conv2d(dense_ch, dense_ch, kernel_size=1),
        #     nn.SiLU(),
        #     nn.Conv2d(dense_ch, dense_ch, kernel_size=1),
        # )

        # ------------------------- GTR 分支（冻结先验） ------------------------- #
        # 如果提供了 GTR 权重路径且已正确安装 GTR 模型，则在 ASAM 中挂载一条“冻结 GTR 分支”
        self.use_gtr = False
        self.gtr = None
        if gtr_ckpt_path is not None and GaussianTruncationRepresentation is not None:
            try:
                # 根据数据集类型选择与 GTR 训练时一致的输入通道数：
                # - LIDC: 原始 GTR 以 1 通道输入训练
                # - ISIC / ISBI: 原始 GTR 以 3 通道 RGB 输入训练
                if self.dataset.lower() in ["isic", "isic3_style_concat", "isbi", "isbi2016"]:
                    gtr_in_channels = 3
                else:
                    gtr_in_channels = self.input_channels

                self.gtr = GaussianTruncationRepresentation(
                    name="asam_gtr_prior",
                    num_channels=gtr_in_channels,
                    num_classes=1,
                )
                ckpt = torch.load(gtr_ckpt_path, map_location="cpu")
                state_dict = ckpt.get("model_state_dict", ckpt)
                self.gtr.load_state_dict(state_dict)
                # 冻结 GTR 参数
                for p in self.gtr.parameters():
                    p.requires_grad = False
                self.gtr.eval()
                self.use_gtr = True
                print(f"[ASAM] Loaded GTR checkpoint from {gtr_ckpt_path}")
            except Exception as e:
                print(f"[ASAM] Failed to load GTR checkpoint from {gtr_ckpt_path}: {e}")
                self.gtr = None
                self.use_gtr = False


    def forward(
        self,
        batch_input,
        batch_input_ori,
        batch_boxori,
        batch_boxshift,
        batch_mask,
        device,
        input_size=128,
        train=True,
    ):
        img_size = input_size
        self.lora_sam.sam.to(device)
        self.prior_object.to(device)
        #self.prior_box.to(device)
        self.posterior_object.to(device)
        #self.posterior_box.to(device)
        self.fcomb_object.to(device)
        #self.fcomb_box.to(device)

        # 如果启用了 GTR 分支，将其也移动到对应设备
        if self.use_gtr and self.gtr is not None:
            self.gtr.to(device)

        input_images = self.lora_sam.sam.preprocess(batch_input)
        image_embeddings = self.lora_sam.sam.image_encoder(input_images)

        # ------------------------- GTR 分支：生成随机先验 mask ------------------------- #
        # 使用训练好的 GTR 作为冻结先验，输出一个随机样本 mask，作为额外的 prompt mask
        gtr_mask_for_prompt = None
        if self.use_gtr and self.gtr is not None:
            with torch.no_grad():
                # 根据 GTR 训练时的输入通道选择合适的输入：
                # - LIDC: 使用单通道原图 batch_input_ori
                # - ISIC / ISBI: 使用 3 通道 RGB 图像 batch_input
                if self.dataset.lower() in ["isic", "isic3_style_concat", "isbi", "isbi2016"]:
                    gtr_input = batch_input.to(device)         # [B,3,H,W]
                else:
                    gtr_input = batch_input_ori.to(device)     # [B,1,H,W]

                gtr_logits, gtr_out, _ = self.gtr(gtr_input)
                # 从分布中采样（随机起点），再经过 sigmoid 得到 [0,1] 概率 mask
                gtr_sample = torch.sigmoid(gtr_out["distribution"].rsample())
                if gtr_sample.dim() == 3:
                    gtr_sample = gtr_sample.unsqueeze(1)  # 保证形状为 Bx1xHxW
                # SAM 的 PromptEncoder 期望 mask 输入大小为 mask_input_size
                mask_input_size = self.lora_sam.sam.prompt_encoder.mask_input_size
                gtr_mask_for_prompt = F.interpolate(
                    gtr_sample,
                    size=mask_input_size,
                    mode="bilinear",
                    align_corners=False,
                )

        # 将 GTR mask + bbox 组合提示注入 prompt_encoder：
        # - masks: 来自冻结 GTR 的随机先验 mask（提供语义先验与多样性）
        # - boxes: 训练时的扰动 bbox（提供位置约束，提升稳定性）
        if gtr_mask_for_prompt is not None:
            # sparse_embeddings_shift, dense_embeddings_shift = self.lora_sam.sam.prompt_encoder(
            #     points=None, boxes=batch_boxshift, masks=gtr_mask_for_prompt
            # )
            sparse_embeddings_shift, dense_embeddings_shift = self.lora_sam.sam.prompt_encoder(
                points=None, boxes=batch_boxshift, masks=gtr_mask_for_prompt
            )
        else:
            sparse_embeddings_shift, dense_embeddings_shift = self.lora_sam.sam.prompt_encoder(
                points=None, boxes=batch_boxshift, masks=None
            )
        sparse_embeddings_ori, dense_embeddings_ori = self.lora_sam.sam.prompt_encoder(
            points=None, boxes=batch_boxori, masks=None
        )
        self.prior_box_latent_space = self.prior_box.forward(image_embeddings, sparse_embeddings_shift)
        self.prior_object_latent_space = self.prior_object.forward(batch_input_ori)

        # ------------ box/object latent 采样与解码（保持原有结构） ------------ #
        if train:
            if self.posterior_fixed_std_normal:
                import pdb; pdb.set_trace()
                bsz = image_embeddings.shape[0]
                device_ = image_embeddings.device
                dtype_ = image_embeddings.dtype
                loc_box = torch.zeros((bsz, self.latent_dim), device=device_, dtype=dtype_)
                scale_box = torch.ones_like(loc_box)
                self.posterior_box_latent_space = Independent(Normal(loc=loc_box, scale=scale_box), 1)

                loc_obj = torch.zeros((bsz, self.latent_dim), device=device_, dtype=dtype_)
                scale_obj = torch.ones_like(loc_obj)
                self.posterior_object_latent_space = Independent(Normal(loc=loc_obj, scale=scale_obj), 1)
            else:
                self.posterior_box_latent_space = self.posterior_box.forward(
                    image_embeddings, sparse_embeddings_ori, sparse_embeddings_shift
                )
                self.posterior_object_latent_space = self.posterior_object.forward(batch_input_ori, batch_mask.unsqueeze(1))

            #self.z_posterior_box = self.posterior_box_latent_space.rsample()
            self.z_posterior_object = self.posterior_object_latent_space.rsample()
            #self.z_prior_box = self.prior_box_latent_space.rsample()
            self.z_prior_object = self.prior_object_latent_space.rsample()
            #dense_embeddings_disturb = self.fcomb_box.forward(dense_embeddings_shift, self.z_posterior_box)
            dense_embeddings_disturb = dense_embeddings_shift
            image_embeddings_disturb = self.fcomb_object.forward(image_embeddings, self.z_posterior_object)
        else:
            #self.z_prior_box = self.prior_box_latent_space.sample()
            self.z_prior_object = self.prior_object_latent_space.sample()
            #dense_embeddings_disturb = self.fcomb_box.forward(dense_embeddings_shift, self.z_prior_box)
            dense_embeddings_disturb = dense_embeddings_shift
            image_embeddings_disturb = self.fcomb_object.forward(image_embeddings, self.z_prior_object)

        # ------------ 扩散分支：对 dense prompt 做正则与采样，并与 z0 融合 ------------ #
        diff_loss = None
        z0 = dense_embeddings_disturb  # 作为“干净” dense prompt

        if train:
            b = z0.shape[0]
            t = torch.randint(
                low=0,
                high=self.prompt_diffusion.timesteps,
                size=(b,),
                device=z0.device,
                dtype=torch.long,
            )
            epsilon = torch.randn_like(z0)
            zt = self.prompt_diffusion.q_sample(z0, t, epsilon)

            if not hasattr(self, "prompt_diffusion_debug_printed"):
                print(
                    f"[ASAM] Prompt diffusion active: "
                    f"z0 shape={tuple(z0.shape)}, "
                    f"t[0]={int(t[0])}, "
                    f"epsilon std={float(epsilon.std().item()):.4f}"
                )
                self.prompt_diffusion_debug_printed = True

            eps_pred = self.prompt_denoiser(zt, image_embeddings_disturb, t)
            diff_loss = F.mse_loss(eps_pred, epsilon)

        # 采样扩散分支的 z0_d
        with torch.no_grad():
            dense_sample = self._prompt_p_sample_loop(
                image_embeddings_disturb,
                steps=100,
                sampler="DDIM",
            )

        # 残差校正：扩散只提供增量，不覆盖 z0
        # delta = self.residual_conv(dense_sample)
        # dense_embeddings_for_decoder = z0 + delta
        dense_embeddings_for_decoder = z0*0.9 + dense_sample*0.1

        low_res_masks, iou_predictions = self.lora_sam.sam.mask_decoder(
            image_embeddings=image_embeddings_disturb,
            image_pe=self.lora_sam.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings_shift,
            dense_prompt_embeddings=dense_embeddings_for_decoder,
            multimask_output=True
        )
        masks = self.lora_sam.sam.postprocess_masks(
            low_res_masks,
            input_size=(img_size,img_size ),
            original_size=(128, 128)
        )
      
        outputs = {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks,
            'diff_loss': diff_loss,
        }
        return outputs

    def _prompt_p_sample_loop(
        self,
        image_embeddings: torch.Tensor,
        steps: int = None,
        sampler: str = "DDIM",
    ) -> torch.Tensor:
        """
        在 prompt latent 空间上执行反向扩散采样：
        - 使用 self.prompt_diffusion 中的 alpha_cumprod / betas
        - 若 sampler='DDPM' 使用随机项 s * epsilon；若为 'DDIM' 则 s=0，得到确定性采样
        """
        device = image_embeddings.device
        B, _, H, W = image_embeddings.shape

        T_total = self.prompt_diffusion.timesteps
        if steps is None or steps > T_total:
            steps = T_total

        alphas_cumprod = self.prompt_diffusion.alphas_cumprod.to(device)
        alphas_cumprod_prev = self.prompt_diffusion.alphas_cumprod_prev.to(device)
        betas = self.prompt_diffusion.betas.to(device)

        dense_ch = self.prompt_denoiser.conv_out.out_channels
        z_t = torch.randn(B, dense_ch, H, W, device=device)  # z_T

        for t_idx in reversed(range(steps)):
            acum = alphas_cumprod[t_idx]
            acum1 = alphas_cumprod_prev[t_idx]
            beta = betas[t_idx]

            t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)

            eps_pred = self.prompt_denoiser(z_t, image_embeddings, t_batch)
            eps = torch.randn_like(z_t)

            if t_idx == 0:
                z_t = (
                    (z_t - torch.sqrt(1.0 - acum) * eps_pred) / torch.sqrt(acum)
                    + torch.sqrt(beta) * eps
                )
            else:
                if sampler == "DDPM":
                    s = torch.sqrt(
                        ((1.0 - acum1) / (1.0 - acum)) * (1.0 - acum / acum1)
                    )
                else:
                    s = torch.tensor(0.0, device=device, dtype=acum.dtype)

                z_t = (
                    torch.sqrt(acum1)
                    * ((z_t - torch.sqrt(1.0 - acum) * eps_pred) / torch.sqrt(acum))
                    + torch.sqrt(1.0 - acum1 - s**2) * eps_pred
                    + s * eps
                )

        return z_t


