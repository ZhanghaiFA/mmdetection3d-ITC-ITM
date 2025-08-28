import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
# 这个是第一篇文章使用的

class LSSNeck(nn.Module):
    """
    简化的LSS专用neck
    - 保持现有接口不变
    - 大幅简化内部结构
    - 专注核心功能：深度概率分布 + 投影特征
    """
    def __init__(self, 
                 in_channels=64, 
                 D=118, 
                 C=80):
        super().__init__()
        self.D = D  # 深度bins数量
        self.C = C  # 特征维度
        self.in_channels = in_channels
        self.out_channels = D + C
        
        # 内置配置参数
        self.mid_channels = 128  # 保持足够的特征表达能力
        self.norm_cfg = dict(type='BN2d', requires_grad=True)
        self.act_cfg = dict(type='ReLU', inplace=False)
        
        self._build_modules()
    
    def _build_modules(self):
        """构建简化的neck结构"""
        
        # 1. 输入特征处理
        self.input_conv = ConvModule(
            self.in_channels,
            self.mid_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            inplace=False
        )
        
        # 2. 简化的特征增强
        self.feature_enhance = SimpleFeatureBlock(
            self.mid_channels,
            self.mid_channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        
        # 3. 深度预测分支
        self.depth_branch = nn.Sequential(
            ConvModule(
                self.mid_channels, self.mid_channels // 2, 3, padding=1,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False
            ),
            nn.Conv2d(self.mid_channels // 2, self.D, 1)
        )
        
        # 4. 特征预测分支
        self.feature_branch = nn.Sequential(
            ConvModule(
                self.mid_channels, self.mid_channels // 2, 3, padding=1,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False
            ),
            nn.Conv2d(self.mid_channels // 2, self.C, 1)
        )
        
        # 5. 残差连接
        self.shortcut = nn.Conv2d(self.in_channels, self.mid_channels, 1) \
                       if self.in_channels != self.mid_channels else nn.Identity()
    
    def forward(self, x):
        """
        简化的前向传播
        
        Args:
            x: [B, 64, H/8, W/8] 深度估计模块的融合特征
        
        Returns:
            [B, (D + C), H/8, W/8] 深度概率分布 + 投影特征
        """
        # 1. 输入处理
        x_conv = self.input_conv(x)
        
        # 2. 特征增强
        x_enhanced = self.feature_enhance(x_conv)
        
        # 3. 残差连接
        shortcut = self.shortcut(x)
        x_final = x_enhanced + shortcut
        
        # 4. 两个分支预测
        depth_out = self.depth_branch(x_final)     # [B, D, H/8, W/8]
        feature_out = self.feature_branch(x_final) # [B, C, H/8, W/8]
        
        # 5. 拼接输出
        output = torch.cat([depth_out, feature_out], dim=1)  # [B, D+C, H/8, W/8]
        
        return output
    
    def forward_separate(self, x):
        """
        返回分离的深度和特征，便于LSS使用
        
        Returns:
            depth_logits: [B, D, H/8, W/8] 深度logits
            feature_maps: [B, C, H/8, W/8] 特征maps
        """
        # 主要处理流程
        x_conv = self.input_conv(x)
        x_enhanced = self.feature_enhance(x_conv)
        
        shortcut = self.shortcut(x)
        x_final = x_enhanced + shortcut
        
        # 分别返回两个分支
        depth_logits = self.depth_branch(x_final)
        feature_maps = self.feature_branch(x_final)
        
        return depth_logits, feature_maps


class SimpleFeatureBlock(nn.Module):
    """简化的特征处理块"""
    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        
        # 主要特征处理 - 只用两个3x3卷积
        self.conv1 = ConvModule(
            in_channels, out_channels, 3, padding=1,
            norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False
        )
        
        self.conv2 = ConvModule(
            out_channels, out_channels, 3, padding=1,
            norm_cfg=norm_cfg, act_cfg=None, inplace=False  # 最后不加激活
        )
        
        # 简单的通道注意力 - 只保留SE
        self.se = SimpleSE(out_channels)
        
        self.final_act = nn.ReLU(inplace=False)
    
    def forward(self, x):
        # 两层卷积
        out = self.conv1(x)
        out = self.conv2(out)
        
        # SE注意力
        out = self.se(out)
        
        # 残差连接
        out = out + x
        out = self.final_act(out)
        
        return out


class SimpleSE(nn.Module):
    """简化的SE注意力"""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 简化：只用一层全连接，减少参数
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# 使用示例和测试
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟输入
    B = 2
    x = torch.randn(B, 64, 36, 64).to(device)
    
    # 创建简化的LSS neck
    lss_neck = LSSNeck(in_channels=64, D=118, C=80).to(device)
    
    print("=== 简化版 LSS Neck 测试 ===")
    print(f"输入特征: {x.shape}")
    
    # 方式1: 拼接输出（兼容现有接口）
    output = lss_neck(x)
    print(f"拼接输出: {output.shape} (应该是 [2, 198, 36, 64])")
    
    # 方式2: 分离输出（便于LSS使用）
    depth_logits, feature_maps = lss_neck.forward_separate(x)
    print(f"深度logits: {depth_logits.shape} (应该是 [2, 118, 36, 64])")
    print(f"特征maps: {feature_maps.shape} (应该是 [2, 80, 36, 64])")
    
    # 参数统计对比
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== 模型对比 ===")
    print(f"简化版参数量: {count_parameters(lss_neck):,}")
    
    # 计算复杂度估算
    from thop import profile, clever_format
    flops, params = profile(lss_neck, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
    
    # 梯度检查
    loss = output.sum()
    loss.backward()
    
    unused_params = []
    for name, param in lss_neck.named_parameters():
        if param.grad is None:
            unused_params.append(name)
    
    if unused_params:
        print(f"❌ 未使用参数: {unused_params}")
    else:
        print("✅ 所有参数都有梯度！")
    
    print(f"\n=== 简化后的架构 ===")
    print(f"输入处理: 64 → 128 (一个3x3卷积)")
    print(f"特征增强: 128 → 128 (两个3x3卷积 + 简化SE)")
    print(f"深度分支: 128 → 64 → 118 (一个3x3 + 一个1x1)")
    print(f"特征分支: 128 → 64 → 80 (一个3x3 + 一个1x1)")
    print(f"残差连接: 64 → 128 (一个1x1卷积)")
    
    print(f"\n=== 使用方法 ===")
    print(f"neck = LSSNeck()  # 接口完全不变")
    print(f"output = neck(x)  # 拼接输出")
    print(f"depth, feat = neck.forward_separate(x)  # 分离输出")