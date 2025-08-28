import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC

##############################################################################
# 1) 部分卷积 + 空洞卷积 (仅适用于 depthwise)
##############################################################################
class PartialDilatedDepthwiseConv(nn.Module):
    """
    depthwise conv + partial conv + dilation
    - stride>1 时下采样 mask
    - 返回 (out, new_mask)
    """
    def __init__(self, channels, kernel_size=3, stride=1, dilation=1, bias=False):
        super().__init__()
        self.channels   = channels
        self.kernel_size= kernel_size
        self.stride     = stride
        self.dilation   = dilation
        self.padding    = (kernel_size -1)//2 * dilation
        self.groups     = channels  # depthwise conv

        # depthwise conv weight => [channels,1,kH,kW]
        self.weight = nn.Parameter(torch.empty(channels,1,kernel_size,kernel_size))
        if bias:
            self.bias= nn.Parameter(torch.zeros(channels))
        else:
            self.bias= None

        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, mask=None):
        """
        x: [B, channels, H, W]
        mask: [B,1,H,W], or None
        returns (out, new_mask)
          out: [B,channels,H/stride,W/stride]
          new_mask: [B,1,H/stride,W/stride], or None
        """
        if mask is None:
            # 普通空洞 depthwise
            out= F.conv2d(
                x, self.weight, self.bias,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            )
            return out, None

        # partial conv
        # 如果 stride>1 => 下采样mask
        new_mask= mask
        if self.stride>1 and (x.shape[2:4] != new_mask.shape[2:4]):
            new_mask = F.max_pool2d(new_mask, kernel_size=self.stride, stride=self.stride)

        # x_valid
        x_valid= x * new_mask  # 这里 mask仍是旧尺寸 => x
        # 统计 mask_sum => stride=1, groups=1
        with torch.no_grad():
            weight_mask= torch.ones((1,1,self.kernel_size,self.kernel_size),
                                    device=x.device,dtype=x.dtype)
            mask_sum = F.conv2d(
                new_mask, weight_mask, None,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=1
            )

        # depthwise => stride => out
        out= F.conv2d(
            x_valid, self.weight, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )

        # 同理 mask_sum 也需要兼容 stride
        if self.stride>1 and (out.shape[2:4] != mask_sum.shape[2:4]):
            mask_sum = F.max_pool2d(mask_sum, kernel_size=self.stride, stride=self.stride)

        eps=1e-6
        mask_sum_clamped= torch.clamp(mask_sum, min=eps)
        out= out * (mask_sum>0).float()
        out= out / mask_sum_clamped
        return out, new_mask

##############################################################################
# 2) InvertedResidual => 返回 (out, new_mask)
##############################################################################
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, dilation_dw=1):
        super().__init__()
        self.stride= stride
        self.use_res_connect= (stride==1 and inp==oup)
        hidden_dim= int(inp*expand_ratio)

        self.pw1= nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True)
        )
        self.dw= PartialDilatedDepthwiseConv(
            channels= hidden_dim,
            kernel_size=3,
            stride=stride,
            dilation=dilation_dw,
            bias=False
        )
        self.dw_bn= nn.BatchNorm2d(hidden_dim)
        self.dw_relu= nn.ReLU(True)

        self.pw2= nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x, mask=None):
        """
        returns (out, new_mask)
        """
        out= self.pw1(x)
        # dw => partial conv
        out, new_mask= self.dw(out, mask=mask)
        out= self.dw_bn(out)
        out= self.dw_relu(out)
        out= self.pw2(out)
        if self.use_res_connect:
            out += x
        return out, new_mask

##############################################################################
# 3) MobileViTBlock
##############################################################################
class MobileViTBlock(nn.Module):
    def __init__(self, in_ch, transformer_dim=64):
        super().__init__()
        self.local_conv1= nn.Conv2d(in_ch, in_ch, 3,1,1,bias=False)
        self.bn1= nn.BatchNorm2d(in_ch)
        self.relu= nn.ReLU(True)

        self.conv_proj_in= nn.Conv2d(in_ch,transformer_dim,1,bias=False)
        self.ln= nn.LayerNorm(transformer_dim)
        self.qkv_proj= nn.Linear(transformer_dim,transformer_dim*3,bias=True)
        self.out_proj= nn.Linear(transformer_dim,transformer_dim,bias=True)
        self.conv_proj_out= nn.Conv2d(transformer_dim, in_ch,1,bias=False)
        self.bn2= nn.BatchNorm2d(in_ch)

    def forward(self, x, mask=None):  # mask 用于可选地屏蔽注意力
        B, C, H, W = x.shape
        y = self.local_conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv_proj_in(y)
        y = y.permute(0, 2, 3, 1).contiguous().view(B, H*W, -1)
        y = self.ln(y)
        qkv = self.qkv_proj(y)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        d = q.shape[-1]
        attn = torch.bmm(q, k.transpose(1, 2)) / (d**0.5)

        # 如果有 mask，则在注意力中屏蔽无效位置
        if mask is not None:
            mask_flat = mask.view(B, -1)            # [B, HW]
            attn_mask = mask_flat.unsqueeze(1)      # [B, 1, HW]
            attn_mask = attn_mask.repeat(1, H*W, 1) # [B, HW, HW]
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
            # 防止全部为 -inf 导致数值问题
            attn = torch.where(
                attn_mask.sum(dim=-1, keepdim=True) == 0,
                torch.zeros_like(attn),
                attn
            )

        attn = torch.softmax(attn, dim=-1)
        z = torch.bmm(attn, v)
        z = self.out_proj(z)

        z = z.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        y = self.conv_proj_out(z)
        y = self.bn2(y)
        return x + y, mask  # 返回输出和原 mask (若有)

##############################################################################
# 4) Guide
##############################################################################
class Guide(nn.Module, ABC):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(True)
        )

    def forward(self, lidar_feat, img_feat, mask=None):
        x = torch.cat([lidar_feat, img_feat], dim=1)
        out = self.conv(x)
        return out * mask if mask is not None else out  # 仅在需要时应用mask

##############################################################################
# 5) Encoder (在最后加一个 3×3 卷积)
##############################################################################
# class Encoder(nn.Module):
class Model(nn.Module):
    """
    输出 (x, mask)，其中 x 在 1/8 尺度。
    这里在输出前加了一个 3×3 卷积(不改变通道数和空间大小)，
    方便你在训练时直接拿这个卷积输出做预测或损失计算等。
    """
    def __init__(self, args, bc=8, dilation_dw=1):
        super().__init__()
        self.args= args
        self.bc= bc
        self.dilation_dw= dilation_dw

        # --- 下采样 => 1/8 ---
        self.img_stage1= InvertedResidual(3, 16, stride=2, expand_ratio=2, dilation_dw=dilation_dw)
        self.img_stage2= InvertedResidual(16,32, stride=2, expand_ratio=2, dilation_dw=dilation_dw)
        self.img_stage3= InvertedResidual(32,64, stride=2, expand_ratio=2, dilation_dw=dilation_dw)

        self.lidar_stage1= InvertedResidual(1, 16, stride=2, expand_ratio=2, dilation_dw=dilation_dw)
        self.lidar_stage2= InvertedResidual(16,32, stride=2, expand_ratio=2, dilation_dw=dilation_dw)
        self.lidar_stage3= InvertedResidual(32,64, stride=2, expand_ratio=2, dilation_dw=dilation_dw)

        self.guide1= Guide(16)
        self.guide2= Guide(32)
        self.guide3= Guide(64)

        # --- MobileViT block ---
        self.img_mvit= MobileViTBlock(64,transformer_dim=64)
        self.lidar_mvit= MobileViTBlock(64,transformer_dim=64)
        self.guide_mvit= Guide(64)

        # *** 在这里添加一个 3x3 卷积，不改变通道数 & 空间分辨率 ***
        # 原本输出通道数是 64，这里保持不变；padding=1 确保输出 H/8,W/8
        # self.final_3x3_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, rgb, depth):
        """
        rgb:   [B,3,H,W]
        depth: [B,1,H,W]
        返回 x, mask_lidar3 (x 大小是 [B,64,H/8,W/8])
        """
        # 初始mask (只要 depth>0 就视为有效)
        mask_lidar= (depth>0).float()

        # stage1 =>1/2
        x_img1, _= self.img_stage1(rgb, mask=None)
        x_lidar1, mask_lidar1= self.lidar_stage1(depth, mask=mask_lidar)
        x_lidar1= self.guide1(x_lidar1, x_img1)

        # stage2 =>1/4
        x_img2, _= self.img_stage2(x_img1, mask=None)
        x_lidar2, mask_lidar2= self.lidar_stage2(x_lidar1, mask=mask_lidar1)
        x_lidar2= self.guide2(x_lidar2, x_img2)

        # stage3 =>1/8
        x_img3, _= self.img_stage3(x_img2, mask=None)
        x_lidar3, mask_lidar3= self.lidar_stage3(x_lidar2, mask=mask_lidar2)
        x_lidar3= self.guide3(x_lidar3, x_img3)

        # mask 也下采样到1/8
        mask_lidar3 = F.max_pool2d(mask_lidar3, kernel_size=2, stride=2)

        # MobileViT
        x_img_m, _ = self.img_mvit(x_img3, mask=None)
        x_lidar_m, mask_lidar3 = self.lidar_mvit(x_lidar3, mask=mask_lidar3)
        x_lidar_m = self.guide_mvit(x_lidar_m, x_img_m)
        x= x_img_m + x_lidar_m

        

        return x, mask_lidar3


import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthNet(nn.Module):
    def __init__(self, in_channels=64, D=118, C=80):
        """
        in_channels=64 ：预训练模型输出的通道数 (例如1/8下采样的特征)
        D=118         ：深度概率通道数
        C=80          ：最终特征通道数
        """
        super().__init__()
        self.D = D
        self.C = C
        # 这里只是示例，可按需增减卷积、BN等结构
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, D + C, kernel_size=1)  # 输出 (D + C) 通道
        )

    def forward(self, x, mask=0):
        """
        x:    [B*N, 64, fH, fW]        # 1/8下采样的特征
        mask: [B*N, 1,  fH, fW]        # 与x同分辨率的mask

        返回:  [B*N, (D + C), fH, fW]   # 只返回最终数值，不再返回mask
        """
        # 将mask应用到特征 x 上（示例做法：逐点相乘）
        # x_masked = x * mask  # 若mask为1/0，表示保留/置零

        # 经过主干卷积
        out = self.main(x)

        # 只返回卷积结果
        return out
