#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速调试脚本 - 保存张量中的图片
在你的代码中直接调用这个函数即可
"""

import torch
from PIL import Image
import os

def quick_save_image(tensor, save_path="./debug_image.png"):
    """
    快速保存张量中的第1个batch的第1张图片
    
    Args:
        tensor: 输入张量，形状为 (B, N, C, H, W)
        save_path: 保存路径
    """
    try:
        # 提取第1个batch的第1张图片
        img = tensor[0, 0, :, :, :]  # (3, 256, 704)
        
        # 移到CPU并转换为numpy
        if img.is_cuda:
            img = img.cpu()
        
        # 处理值范围
        if img.dtype == torch.float32 or img.dtype == torch.float64:
            if img.min() < 0:
                img = (img + 1) / 2  # [-1,1] -> [0,1]
            elif img.max() > 1:
                img = (img - img.min()) / (img.max() - img.min())  # 归一化
            img = (img * 255).clamp(0, 255).to(torch.uint8)
        
        # 调整通道顺序
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)  # (H, W, C)
        
        # 保存图片
        pil_img = Image.fromarray(img.numpy())
        pil_img.save(save_path)
        print(f"图片已保存到: {save_path}")
        
    except Exception as e:
        print(f"保存图片时出错: {e}")

# 在你的代码中使用示例:
# 在 BEVNoSwin.py 的第259行附近添加:
# 
# from quick_debug_save import quick_save_image
# 
# # 在 x.size() 之后添加:
# B, N, C, H, W = x.size()    # 4 6 3 256 704
# quick_save_image(x, "./first_batch_first_image.png")
