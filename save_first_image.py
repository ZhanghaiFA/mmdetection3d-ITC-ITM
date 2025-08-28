#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
保存张量中第1个batch的第1张图片的调试脚本
张量形状: (B, N, C, H, W) = (4, 6, 3, 256, 704)
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

def save_first_image(tensor, output_dir="./debug_images", filename="first_image.png"):
    """
    保存张量中第1个batch的第1张图片
    
    Args:
        tensor: 输入张量，形状为 (B, N, C, H, W)
        output_dir: 输出目录
        filename: 输出文件名
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查张量形状
    if len(tensor.shape) != 5:
        print(f"错误：张量维度应该是5，但得到了{tensor.shape}")
        return
    
    B, N, C, H, W = tensor.shape
    print(f"张量形状: B={B}, N={N}, C={C}, H={H}, W={W}")
    
    # 提取第1个batch的第1张图片 (索引 [0, 0, :, :, :])
    first_image = tensor[0, 0, :, :, :]  # 形状: (3, 256, 704)
    print(f"提取的图片形状: {first_image.shape}")
    
    # 检查数据类型和值范围
    print(f"数据类型: {first_image.dtype}")
    print(f"值范围: [{first_image.min():.4f}, {first_image.max():.4f}]")
    
    # 如果张量在GPU上，移到CPU
    if first_image.is_cuda:
        first_image = first_image.cpu()
        print("张量已从GPU移动到CPU")
    
    # 转换为PIL图像
    if first_image.dtype == torch.float32 or first_image.dtype == torch.float64:
        # 如果是浮点数，假设值范围在[0,1]或[-1,1]
        if first_image.min() < 0:
            # 值范围可能是[-1,1]，需要转换到[0,1]
            first_image = (first_image + 1) / 2
            print("检测到值范围[-1,1]，已转换到[0,1]")
        elif first_image.max() > 1:
            # 值范围可能超过[0,1]，需要归一化
            first_image = (first_image - first_image.min()) / (first_image.max() - first_image.min())
            print("检测到值范围超出[0,1]，已归一化")
        
        # 转换到[0,255]的uint8
        first_image = (first_image * 255).clamp(0, 255).to(torch.uint8)
    
    # 确保通道顺序是 (H, W, C) 用于PIL
    if first_image.shape[0] == 3:  # 如果通道在第一维
        first_image = first_image.permute(1, 2, 0)  # (H, W, C)
    
    # 转换为numpy数组
    image_np = first_image.numpy()
    
    # 创建PIL图像
    pil_image = Image.fromarray(image_np)
    
    # 保存图像
    output_path = os.path.join(output_dir, filename)
    pil_image.save(output_path)
    print(f"图片已保存到: {output_path}")
    
    # 显示图像信息
    print(f"图片尺寸: {pil_image.size}")
    print(f"图片模式: {pil_image.mode}")
    
    return output_path

def save_all_batch_images(tensor, output_dir="./debug_images"):
    """
    保存所有batch的第1张图片
    
    Args:
        tensor: 输入张量，形状为 (B, N, C, H, W)
        output_dir: 输出目录
    """
    B, N, C, H, W = tensor.shape
    
    for b in range(B):
        filename = f"batch_{b}_first_image.png"
        save_first_image(tensor[b:b+1, :, :, :, :], output_dir, filename)

def create_dummy_tensor_for_testing():
    """
    创建一个测试用的虚拟张量
    """
    # 创建一个形状为 (4, 6, 3, 256, 704) 的虚拟张量
    dummy_tensor = torch.randn(4, 6, 3, 256, 704)
    
    # 将值范围调整到[0,1]以便可视化
    dummy_tensor = torch.sigmoid(dummy_tensor)
    
    return dummy_tensor

if __name__ == "__main__":
    print("=== 张量图片保存调试脚本 ===\n")
    
    # 示例1: 使用虚拟张量测试
    print("1. 使用虚拟张量测试:")
    dummy_tensor = create_dummy_tensor_for_testing()
    save_first_image(dummy_tensor, "./debug_images", "dummy_first_image.png")
    
    print("\n" + "="*50 + "\n")
    
    # 示例2: 保存所有batch的第1张图片
    print("2. 保存所有batch的第1张图片:")
    save_all_batch_images(dummy_tensor, "./debug_images")
    
    print("\n脚本执行完成！")
    print("请检查 ./debug_images/ 目录中的输出图片")
    
    # 使用说明
    print("\n=== 使用说明 ===")
    print("1. 将此脚本放在你的项目目录中")
    print("2. 修改脚本中的tensor变量为你的实际张量")
    print("3. 运行脚本: python save_first_image.py")
    print("4. 或者在你的代码中调用: save_first_image(your_tensor)")
