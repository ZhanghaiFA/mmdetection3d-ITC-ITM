
# projects/BEVFusion/bevfusion/contrastive_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from typing import Optional
import torch.distributed as dist
import numpy as np


@MODELS.register_module() # 如果需要，可以取消注释
class MatchingHeadWithAllInBatchNegatives(nn.Module):
    """
    一个使用批内所有其他样本作为负样本的对比学习匹配头。
    This approach is inspired by CLIP and is more powerful than 1-to-1 negative sampling.
    """
    def __init__(
        self,
        in_channels_img: int,
        in_channels_lidar: int,
        proj_dim: int = 256,
        init_logit_scale: float = np.log(1 / 0.07) # 来自CLIP的经典初始化
    ):
        """
        Args:
            in_channels_img: 输入图像特征的通道数
            in_channels_lidar: 输入LiDAR特征的通道数
            proj_dim: 投影到的共享嵌入空间的维度
            init_logit_scale: 可学习的温度系数的初始值
        """
        super().__init__()

        # 为两个模态创建独立的投影器，将它们映射到同一个维度的嵌入空间
        self.proj_img = nn.Linear(in_channels_img, proj_dim)
        self.proj_lidar = nn.Linear(in_channels_lidar, proj_dim)
        
        # 可学习的温度系数，用于缩放logits，控制softmax的锐利程度，对稳定训练至关重要
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        
        # 损失函数使用交叉熵，因为这是一个多分类问题
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, lidar_feat: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lidar_feat: [B, C_lidar] LiDAR特征向量
            img_feat:   [B, C_img] 图像特征向量
        """
        B = lidar_feat.size(0)
        
        # 1. 将特征投影到共享嵌入空间
        zL = self.proj_lidar(lidar_feat)  # [B, proj_dim]
        zI = self.proj_img(img_feat)      # [B, proj_dim]

        # 2. L2归一化特征。这是计算余弦相似度的关键步骤。
        zL = F.normalize(zL, p=2, dim=1)
        zI = F.normalize(zI, p=2, dim=1)

        # 3. 计算所有LiDAR特征与所有图像特征之间的余弦相似度
        # zL @ zI.T 的结果是一个 [B, B] 的矩阵，其中 similarity_matrix[i, j] 
        # 是第i个LiDAR特征和第j个图像特征的相似度。
        similarity_matrix = torch.matmul(zL, zI.T)

        # 4. 应用可学习的温度系数
        # .exp() 是因为我们通常在对数空间中学习这个参数以保证其为正
        logit_scale = self.logit_scale.exp()
        logits = similarity_matrix * logit_scale

        # 5. 创建标签。对于第i个LiDAR特征，其正确的匹配是第i个图像特征。
        # 因此，标签是一个从0到B-1的序列。
        labels = torch.arange(B, device=lidar_feat.device, dtype=torch.long)

        # 6. 计算对称损失
        # loss_l2i: 以LiDAR为锚点，在所有图像中找到匹配项
        loss_l2i = self.criterion(logits, labels)
        
        # loss_i2l: 以图像为锚点，在所有LiDAR中找到匹配项（使用转置的logits矩阵）
        loss_i2l = self.criterion(logits.T, labels)
        
        # 最终损失是两者之和或平均，这使得学习过程更加稳定
        total_loss = (loss_l2i + loss_i2l) / 2.0
        
        return total_loss


def _mlp(in_dim: int, out_dim: int):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
        nn.Linear(out_dim, out_dim)
    )

@MODELS.register_module()
class MatchingHead(nn.Module):
    """
    ITM（二分类）：判别 (z_L, z_I) 是否匹配
    只能在当前单机单卡的batch中使用，即使使用了accumulative_counts=3也没有用
    """
    def __init__(
        self,
        in_channels_img: int,          # e.g. 80
        in_channels_lidar: int,        # e.g. 256
        hidden: int = 256,
        use_projector: bool = True,
        proj_dim: int = 256
    ):
        super().__init__()
        self.use_projector = use_projector

        if use_projector:
            self.proj_img = _mlp(in_channels_img, proj_dim)
            self.proj_lidar = _mlp(in_channels_lidar, proj_dim)
            concat_dim = 2 * proj_dim
        else:
            # 不使用投影头时，直接拼接原始 pooled 向量
            self.proj_img = nn.Identity()
            self.proj_lidar = nn.Identity()
            concat_dim = in_channels_img + in_channels_lidar

        self.net = nn.Sequential(
            nn.Linear(concat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)  # 输出 logits
        )
        self.criterion = nn.BCEWithLogitsLoss()

    @staticmethod
    def _shuffle_except_diag(x: torch.Tensor) -> torch.Tensor:
        """打乱顺序，避免 perm[i] == i；B==1 时做退化处理"""
        B = x.size(0)
        if B == 1:
            # 没有负样本可构造，返回自身（上层可选择跳过 ITM 或权重置零）
            return x
        perm = torch.randperm(B, device=x.device)
        eq_mask = perm.eq(torch.arange(B, device=x.device))
        if eq_mask.any():
            # 将与原索引相等的位置循环右移一位
            perm[eq_mask] = (perm[eq_mask] + 1) % B
        return x[perm]

    def forward(self, lidar_feat: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
        """
        输入:
          - lidar_feat: [B, C_lidar] （来自 BEV GAP）
          - img_feat  : [B, C_img]   （来自 BEV GAP）
        输出:
          - loss_itm (scalar tensor)
        """
        # 两侧各自 projector
        zL = self.proj_lidar(lidar_feat)  # [B, D] or [B, C_lidar]
        zI = self.proj_img(img_feat)      # [B, D] or [B, C_img]

        # 正样本配对
        pos_pairs = torch.cat([zL, zI], dim=1)  # [B, *]

        # 负样本（打乱图像向量）
        neg_img = self._shuffle_except_diag(zI)
        neg_pairs = torch.cat([zL, neg_img], dim=1)  # [B, *]

        pairs = torch.cat([pos_pairs, neg_pairs], dim=0)  # [2B, *]
        labels = torch.cat([
            torch.ones(zL.size(0), 1, device=zL.device),
            torch.zeros(zL.size(0), 1, device=zL.device)
        ], dim=0)

        logits = self.net(pairs).float()  # [2B,1]
        loss = self.criterion(logits, labels)
        return loss



class FocalBCEWithLogitsLoss(nn.Module):
    """
    Focal Loss for binary classification with logits.
    Helps the model focus on hard negatives/positives in contrastive learning.
    """
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, 1] raw logits
            targets: [N, 1] binary targets (0 or 1)
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Compute BCE loss components
        log_probs = F.logsigmoid(logits)
        log_one_minus_probs = F.logsigmoid(-logits)
        
        # Focal weight: (1-p)^gamma for positive class, p^gamma for negative class
        pos_weight = (1 - probs) ** self.gamma
        neg_weight = probs ** self.gamma
        
        # Combine with alpha weighting
        pos_loss = -self.alpha * pos_weight * log_probs
        neg_loss = -(1 - self.alpha) * neg_weight * log_one_minus_probs
        
        # Select appropriate loss based on target
        focal_loss = targets * pos_loss + (1 - targets) * neg_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


@MODELS.register_module()
class SimpleMatchingHeadWithFocalLoss(nn.Module):
    """
    Minimal change version - just replace BCE with Focal BCE
    """
    def __init__(
        self,
        in_channels_img: int,
        in_channels_lidar: int,
        hidden: int = 256,
        use_projector: bool = True,
        proj_dim: int = 256,
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.use_projector = use_projector

        if use_projector:
            self.proj_img = _mlp(in_channels_img, proj_dim)
            self.proj_lidar = _mlp(in_channels_lidar, proj_dim)
            concat_dim = 2 * proj_dim
        else:
            self.proj_img = nn.Identity()
            self.proj_lidar = nn.Identity()
            concat_dim = in_channels_img + in_channels_lidar

        self.net = nn.Sequential(
            nn.Linear(concat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )
        
        # Use Focal BCE instead of standard BCE
        self.criterion = FocalBCEWithLogitsLoss(alpha=focal_alpha, gamma=focal_gamma)

    @staticmethod
    def _shuffle_except_diag(x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        if B == 1:
            return x
        perm = torch.randperm(B, device=x.device)
        eq_mask = perm.eq(torch.arange(B, device=x.device))
        if eq_mask.any():
            perm[eq_mask] = (perm[eq_mask] + 1) % B
        return x[perm]

    def forward(self, lidar_feat: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
        zL = self.proj_lidar(lidar_feat)
        zI = self.proj_img(img_feat)

        pos_pairs = torch.cat([zL, zI], dim=1)
        neg_img = self._shuffle_except_diag(zI)
        neg_pairs = torch.cat([zL, neg_img], dim=1)

        pairs = torch.cat([pos_pairs, neg_pairs], dim=0)
        labels = torch.cat([
            torch.ones(zL.size(0), 1, device=zL.device),
            torch.zeros(zL.size(0), 1, device=zL.device)
        ], dim=0)

        logits = self.net(pairs).float()
        loss = self.criterion(logits, labels)
        return loss

@MODELS.register_module()
class FeatureQueue(nn.Module):
    """GPU 上的环形特征队列（register_buffer），支持 DDP 同步初始化"""
    def __init__(self, feature_dim: int, queue_size: int = 16384):
        super().__init__()
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.register_buffer("queue", torch.zeros(queue_size, feature_dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long))  # 已填充数量

    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor):
        # feats: [B, C] 已经 L2 normalize & detach
        B = feats.size(0)
        ptr = int(self.ptr.item())
        end = ptr + B
        if end <= self.queue_size:
            self.queue[ptr:end].copy_(feats)
        else:
            first = self.queue_size - ptr
            self.queue[ptr:].copy_(feats[:first])
            self.queue[:end % self.queue_size].copy_(feats[first:])
        self.ptr[0] = end % self.queue_size
        self.filled[0] = torch.clamp(self.filled + B, max=self.queue_size)

    @torch.no_grad()
    def sample_negatives(self, k: int, device=None) -> torch.Tensor:
        n = int(self.filled.item())
        if n == 0:
            # 队列空时，返回随机向量（归一化）
            rand = torch.randn(k, self.feature_dim, device=device)
            return F.normalize(rand, dim=1)
        idx = torch.randint(0, n, (k,), device=self.queue.device)
        neg = self.queue[idx]
        return neg.to(device) if device and neg.device != device else neg

    def __len__(self):
        return int(self.filled.item())

@MODELS.register_module()
class ContrastiveHead(nn.Module):
    """
    LiDAR ↔ Image InfoNCE（ITC）：
      - 各自 projector（支持两模态通道数不同）
      - 支持队列负样本 +（可选）DDP all_gather 的 in-batch negatives
      - 屏蔽 in-batch 中的“自身正样本”对角线，避免泄漏
    """
    def __init__(self,
                 in_channels_img: int,      # e.g. 80
                 in_channels_lidar: int,    # e.g. 256
                 proj_dim: int = 256,
                 queue_size: int = 16384,
                 temperature: float = 0.07,
                 neg_k: int = 4096,
                 use_all_gather: bool = False):
        super().__init__()
        self.proj_img = _mlp(in_channels_img, proj_dim)
        self.proj_lidar = _mlp(in_channels_lidar, proj_dim)
        self.queue = FeatureQueue(feature_dim=proj_dim, queue_size=queue_size)
        self.neg_k = neg_k
        self.use_all_gather = use_all_gather
        # 可学习温度（更稳）：logit_scale = exp(param)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def _gather_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        """DDP 下收集所有 rank 的 keys；单卡/未初始化则直接返回本地 x。"""
        if not self.use_all_gather or (not dist.is_available()) or (not dist.is_initialized()):
            return x
        world = dist.get_world_size()
        xs = [torch.zeros_like(x) for _ in range(world)]
        dist.all_gather(xs, x.detach())
        return torch.cat(xs, dim=0)  # [B*world, D]

    def _mask_self_matches(self, neg_in: torch.Tensor, B: int):
        """
        将 in-batch negatives 中“自身正样本”所在的对角元素置为 -inf，避免被当作负样本。
        - 单卡：neg_in 形状 [B, B]
        - DDP：neg_in 形状 [B, B*world]，仅屏蔽本 rank 段的对角
        """
        if not self.use_all_gather or (not dist.is_available()) or (not dist.is_initialized()):
            # 单卡：直接屏蔽对角线
            idx = torch.arange(B, device=neg_in.device)
            neg_in[idx, idx] = float('-inf')
            return

        world = dist.get_world_size()
        rank = dist.get_rank()
        start = rank * B
        end = (rank + 1) * B
        # 只在本 rank 的列区间屏蔽对角
        view = neg_in[:, start:end]          # [B, B]
        idx = torch.arange(B, device=neg_in.device)
        view[idx, idx] = float('-inf')

    def forward(self, lidar_feat: torch.Tensor, img_feat: torch.Tensor):
        """
        输入：
          - lidar_feat: [B, C_lidar]  （来自 BEV GAP）
          - img_feat  : [B, C_img]    （来自 BEV GAP）
        返回：
          - InfoNCE 损失（标量 tensor）
        """
        # 1) projector + L2Norm
        q = F.normalize(self.proj_lidar(lidar_feat), dim=1)  # [B, D]
        k = F.normalize(self.proj_img(img_feat), dim=1)      # [B, D]

        # 2) in-batch negatives（可选 all_gather）
        with torch.no_grad():
            k_all = self._gather_if_needed(k)                # [B*world, D] 或 [B, D]

        # 3) 队列负样本
        neg_mem = self.queue.sample_negatives(self.neg_k, device=q.device)  # [K, D]
        neg_mem = F.normalize(neg_mem, dim=1)  # 保险起见再归一化

        # 4) 相似度构造
        pos = torch.sum(q * k, dim=1, keepdim=True)                   # [B, 1]
        neg_in = q @ k_all.t()                                        # [B, B*world] or [B, B]
        self._mask_self_matches(neg_in, B=q.size(0))                  # 屏蔽对角线
        neg_mem_logits = q @ neg_mem.t()                              # [B, K]

        logits = torch.cat([pos, neg_in, neg_mem_logits], dim=1)      # [B, 1 + B*world + K]
        logits = logits * self.logit_scale.exp().clamp(max=100)

        labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)  # 第一列为正

        loss = F.cross_entropy(logits, labels)

        # 5) 更新队列（入库 image keys）
        with torch.no_grad():
            self.queue.enqueue(k.detach())
        return loss