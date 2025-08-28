# 张量图片保存调试脚本使用说明

## 概述
这些脚本用于调试和可视化你的张量数据，特别是形状为 `(B, N, C, H, W) = (4, 6, 3, 256, 704)` 的图像张量。

## 文件说明

### 1. `save_first_image.py` - 完整功能脚本
- 包含完整的图片保存功能
- 支持多种数据类型和值范围
- 可以保存所有batch的图片
- 包含详细的调试信息输出

### 2. `quick_debug_save.py` - 快速调试脚本
- 轻量级版本，适合快速集成到现有代码中
- 只包含核心的图片保存功能
- 错误处理更简单

## 使用方法

### 方法1: 直接运行脚本测试
```bash
# 测试虚拟张量
python save_first_image.py
```

### 方法2: 在你的代码中集成
在你的 `BEVNoSwin.py` 文件中添加以下代码：

```python
# 在文件顶部导入
from quick_debug_save import quick_save_image

# 在第259行附近添加
def extract_img_feat(self, x, points, lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix, img_metas) -> torch.Tensor:
    B, N, C, H, W = x.size()    # 4 6 3 256 704
    
    # 添加这行来保存图片
    quick_save_image(x, "./debug_first_image.png")
    
    x = x.view(B * N, C, H, W).contiguous()
    # ... 其余代码 ...
```

### 方法3: 在特定位置保存图片
```python
# 保存第1个batch的第1张图片
quick_save_image(x, "./first_batch_first_image.png")

# 保存第2个batch的第1张图片
quick_save_image(x[1:2, :, :, :, :], "./second_batch_first_image.png")

# 保存所有batch的第1张图片
for i in range(x.shape[0]):
    quick_save_image(x[i:i+1, :, :, :, :], f"./batch_{i}_first_image.png")
```

## 输出说明

脚本会：
1. 自动创建 `debug_images` 目录（如果不存在）
2. 将图片保存为PNG格式
3. 自动处理GPU/CPU张量转换
4. 自动处理不同的值范围（[-1,1], [0,1], 或其他范围）
5. 输出详细的调试信息

## 注意事项

1. **数据类型**: 脚本会自动检测张量的数据类型和值范围
2. **GPU支持**: 如果张量在GPU上，会自动移动到CPU进行保存
3. **通道顺序**: 自动处理PyTorch的 (C, H, W) 到PIL的 (H, W, C) 转换
4. **值范围**: 自动处理不同的值范围，确保图片正确显示

## 故障排除

如果遇到问题：
1. 检查张量形状是否正确 (应该是5维)
2. 检查是否有足够的磁盘空间
3. 检查PIL和torch是否正确安装
4. 查看控制台输出的错误信息

## 依赖要求

```bash
pip install torch torchvision pillow numpy
```

## 示例输出

运行脚本后，你应该能看到类似这样的输出：
```
张量形状: B=4, N=6, C=3, H=256, W=704
提取的图片形状: torch.Size([3, 256, 704])
数据类型: torch.float32
值范围: [0.0000, 1.0000]
图片已保存到: ./debug_images/first_image.png
图片尺寸: (704, 256)
图片模式: RGB
```
