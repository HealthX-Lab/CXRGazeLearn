downsample = 10
H = (int(3056 / downsample / 32) + 1) * 32 * downsample
W = (int(2544 / downsample / 32) + 1) * 32 * downsample

# Recalculate dimensions after downsampling
H_downsampled = (H + 1) // downsample
W_downsampled = (W + 1) // downsample

print(H_downsampled, W_downsampled)

import torch
import torchvision.models as models

# 实例化一个DenseNet模型
model = models.densenet201(pretrained=True)

# 创建一个随机输入，大小为(1, 3, 224, 224)，1是批量大小，3是颜色通道数，224x224是图像尺寸
# 这是DenseNet期望的输入大小
x = torch.randn(1, 3, 640, 512)

# 空列表来存储特征图的通道数
features_channels = []

# 我们将关闭梯度计算，因为我们只是在推断
with torch.no_grad():
    # 我们将模型设置为评估模式
    model.eval()
    # 逐层遍历模型
    for name, layer in model.features.named_children():
        # 向前传播
        x = layer(x)
        # 打印层的名称和特征图的形状
        print(f"Layer: {name}, Output shape: {x.shape}")
        # 添加特征图的通道数到列表
        features_channels.append(x.shape[1])


