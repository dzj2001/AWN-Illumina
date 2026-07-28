import torch
import torch.nn as nn
import torch.nn.modules.utils as torch_utils
from collections import namedtuple
ConvLayerConfig = namedtuple('LayerConfig', 'in_channels out_channels kernel_size padding pool')

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 修复关键点：确保中间通道数至少为1
        mid_channels = max(1, in_channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid_channels),  # 输入维度修正
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels)  # 输出维度保持原状
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        scale = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        mask = self.sigmoid(self.conv(combined))
        return x * mask
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 pool=True, relu=True, bn=False):
        super(Conv, self).__init__()
        # 原始卷积模块
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        # 注意力模块（使用保护后的通道数）
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention()
    def forward(self, x):
        x = self.layers(x)
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
# 修正后的残差模块（移除最后 ReLU）
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super(Residual, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        self.ca = ChannelAttention(out_channels, reduction_ratio)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        out = self.relu(self.bn3(out))
        out = self.conv3(out)
        out = self.ca(out)
        out = self.sa(out)
        out += residual
        return out

# 沙漏模块（Hourglass）
class Hourglass(nn.Module):
    def __init__(self, depth, nc, expansion):
        super(Hourglass, self).__init__()
        self.depth = depth
        nc_expanded = nc + expansion
        self.up1 = Residual(nc, nc)
        self.pool = nn.MaxPool2d(2, 2)
        self.low1 = Residual(nc, nc_expanded)
        if self.depth > 1:
            self.low2 = Hourglass(self.depth - 1, nc_expanded, expansion)
        else:
            self.low2 = Residual(nc_expanded, nc_expanded)
        self.low3 = Residual(nc_expanded, nc)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        pool = self.pool(x)
        low1 = self.low1(pool)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up1 = self.up1(x)
        up2 = self.up2(low3)
        return up1 + up2
# 沙漏主干网络 (HourglassBackbone)
class HourglassBackbone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HourglassBackbone, self).__init__()
        self.layers = nn.Sequential(Conv(in_channels, 64, kernel_size=7, stride=2, pool=False,
                                         padding=3, relu=True, bn=True),
                                    Residual(64, 128),
                                    nn.MaxPool2d(2, 2),
                                    Residual(128, 128),
                                    Residual(128, out_channels))
    def forward(self, x):
        return self.layers(x)
 # 精炼主干网络 (RefineBackbone/RefineBackboneKP)
class RefineBackbone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefineBackbone, self).__init__()
        self.layers = nn.Sequential(Conv(in_channels=in_channels, out_channels=6, kernel_size=5, padding=0, pool=True),
                                    Conv(in_channels=6, out_channels=out_channels, kernel_size=5, padding=0, pool=True))

    def forward(self, x):
        return self.layers(x)

 # 精炼主干网络 (RefineBackbone/RefineBackboneKP)
class RefineBackboneKP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefineBackboneKP, self).__init__()
        self.layers = nn.Sequential(Conv(in_channels=in_channels, out_channels=6, kernel_size=5, padding=0, pool=True),
                                    Conv(in_channels=6, out_channels=16, kernel_size=5, padding=0, pool=True),
                                    Conv(in_channels=16, out_channels=out_channels, kernel_size=5, padding=0, pool=True))

    def forward(self, x):
        return self.layers(x)

# 根据配置列表批量生成卷积模块链
def conv_module(layer_configs):
    layers = []
    for layer_config in layer_configs:
        layers.append(Conv(in_channels=layer_config.in_channels,
                           out_channels=layer_config.out_channels,
                           kernel_size=layer_config.kernel_size,
                           padding=layer_config.padding,
                           pool=layer_config.pool))
    return nn.Sequential(*layers)

# 构建分阶段卷积模块（如多尺度处理）
def staged_conv_module(staged_layer_configs):
    stage2layers = []
    for layer_configs, count in staged_layer_configs:
        for _ in range(count):
            stage2layers.append(conv_module(layer_configs))
    return nn.ModuleList(stage2layers)


def fc_module(init_in_features, final_out_features, inner_layer_dims, relu=True):
    layers = []
    for i in range(len(inner_layer_dims)):
        in_features = init_in_features if i == 0 else inner_layer_dims[i - 1]
        out_features = final_out_features if i == (len(inner_layer_dims) - 1) else inner_layer_dims[i]
        layers.append(nn.Linear(in_features=in_features, out_features=out_features))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        # dropout
    return nn.Sequential(*layers)


# noinspection PyProtectedMember
def get_conv2d_layer_output_shape(in_dim, kernel_size, stride, padding, dilation=1):
    in_dim = torch_utils._pair(in_dim)
    kernel_size = torch_utils._pair(kernel_size)
    stride = torch_utils._pair(stride)
    padding = torch_utils._pair(padding)
    dilation = torch_utils._pair(dilation)
    out_dim_0 = (in_dim[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    out_dim_1 = (in_dim[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    return out_dim_0, out_dim_1


def get_conv_module_output_shape(input_dim, module):
    dim = input_dim
    for module_layer in module:
        if isinstance(module_layer, Conv):
            for layer in module_layer.layers:
                if isinstance(layer, nn.Conv2d):
                    dim = get_conv2d_layer_output_shape(dim, layer.kernel_size, layer.stride, layer.padding)
                elif isinstance(layer, nn.MaxPool2d):
                    dim = (int(dim[0] / 2),
                           int(dim[1] / 2))
    return dim

# import torch
# import torch.nn as nn
# import torch.nn.modules.utils as torch_utils
# from collections import namedtuple
#
# ConvLayerConfig = namedtuple('LayerConfig', 'in_channels out_channels kernel_size padding pool')
#
# # 基础卷积模块（Conv）
# class Conv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, pool=True, relu=True, bn=False):
#         super(Conv, self).__init__()
#         layers = [nn.Conv2d(in_channels=in_channels,
#                             out_channels=out_channels,
#                             kernel_size=kernel_size,
#                             stride=stride,
#                             padding=padding)]
#         if pool:
#             layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
#         if bn:
#             layers.append(nn.BatchNorm2d(out_channels))
#         if relu:
#             layers.append(nn.ReLU(inplace=True))
#         self.layers = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.layers(x)
#
# # 定义通道注意力模块（如SE模块）
# #
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction_ratio, in_channels)  # 移除 Sigmoid
#         )
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))  # 输出未归一化
#         max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
#         out = avg_out + max_out
#         out = torch.sigmoid(out).unsqueeze(2).unsqueeze(3)  # 仅此处应用 Sigmoid
#         return x * out.expand_as(x)
# # 在残差模块（Residual）中添加通道注意力
# class Residual(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction_ratio=16):
#         super(Residual, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
#         self.bn2 = nn.BatchNorm2d(out_channels // 2)
#         self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(out_channels // 2)
#         self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
#         self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
#         self.relu = nn.ReLU()
#         # 添加通道注意力模块，通道数为 out_channels
#         self.ca = ChannelAttention(out_channels, reduction_ratio)
#
#     def forward(self, x):
#         residual = x if self.skip is None else self.skip(x)
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn3(out)
#         out = self.relu(out)
#         out = self.conv3(out)
#         # 在残差路径末端应用通道注意力，再与跳跃连接相加
#         out = self.ca(out)
#         out += residual
#         return out
# # 沙漏模块（Hourglass）
# class Hourglass(nn.Module):
#     def __init__(self, depth, nc, expansion):
#         super(Hourglass, self).__init__()
#         self.depth = depth
#         nc_expanded = nc + expansion
#         self.up1 = Residual(nc, nc)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.low1 = Residual(nc, nc_expanded)
#         if self.depth > 1:
#             self.low2 = Hourglass(self.depth - 1, nc_expanded, expansion)
#         else:
#             self.low2 = Residual(nc_expanded, nc_expanded)
#         self.low3 = Residual(nc_expanded, nc)
#         self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
#
#
#     def forward(self, x):
#         pool = self.pool(x)
#         low1 = self.low1(pool)
#         low2 = self.low2(low1)
#         low3 = self.low3(low2)
#         up1 = self.up1(x)
#         up2 = self.up2(low3)
#         return up1 + up2
#
#
# # 沙漏主干网络 (HourglassBackbone)
# class HourglassBackbone(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(HourglassBackbone, self).__init__()
#         self.layers = nn.Sequential(Conv(in_channels, 64, kernel_size=7, stride=2, pool=False,
#                                          padding=3, relu=True, bn=True),
#                                     Residual(64, 128),
#                                     nn.MaxPool2d(2, 2),
#                                     Residual(128, 128),
#                                     Residual(128, out_channels))
#
#     def forward(self, x):
#         return self.layers(x)
#
#
#  # 精炼主干网络 (RefineBackbone/RefineBackboneKP)
# class RefineBackbone(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(RefineBackbone, self).__init__()
#         self.layers = nn.Sequential(Conv(in_channels=in_channels, out_channels=6, kernel_size=5, padding=0, pool=True),
#                                     Conv(in_channels=6, out_channels=out_channels, kernel_size=5, padding=0, pool=True))
#
#     def forward(self, x):
#         return self.layers(x)
#
#  # 精炼主干网络 (RefineBackbone/RefineBackboneKP)
# class RefineBackboneKP(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(RefineBackboneKP, self).__init__()
#         self.layers = nn.Sequential(Conv(in_channels=in_channels, out_channels=6, kernel_size=5, padding=0, pool=True),
#                                     Conv(in_channels=6, out_channels=16, kernel_size=5, padding=0, pool=True),
#                                     Conv(in_channels=16, out_channels=out_channels, kernel_size=5, padding=0, pool=True))
#
#     def forward(self, x):
#         return self.layers(x)
#
# # 根据配置列表批量生成卷积模块链
# def conv_module(layer_configs):
#     layers = []
#     for layer_config in layer_configs:
#         layers.append(Conv(in_channels=layer_config.in_channels,
#                            out_channels=layer_config.out_channels,
#                            kernel_size=layer_config.kernel_size,
#                            padding=layer_config.padding,
#                            pool=layer_config.pool))
#     return nn.Sequential(*layers)
#
# # 构建分阶段卷积模块（如多尺度处理）
# def staged_conv_module(staged_layer_configs):
#     stage2layers = []
#     for layer_configs, count in staged_layer_configs:
#         for _ in range(count):
#             stage2layers.append(conv_module(layer_configs))
#     return nn.ModuleList(stage2layers)
#
#
# def fc_module(init_in_features, final_out_features, inner_layer_dims, relu=True):
#     layers = []
#     for i in range(len(inner_layer_dims)):
#         in_features = init_in_features if i == 0 else inner_layer_dims[i - 1]
#         out_features = final_out_features if i == (len(inner_layer_dims) - 1) else inner_layer_dims[i]
#         layers.append(nn.Linear(in_features=in_features, out_features=out_features))
#         if relu:
#             layers.append(nn.ReLU(inplace=True))
#         # dropout
#     return nn.Sequential(*layers)
#
#
# # noinspection PyProtectedMember
# def get_conv2d_layer_output_shape(in_dim, kernel_size, stride, padding, dilation=1):
#     in_dim = torch_utils._pair(in_dim)
#     kernel_size = torch_utils._pair(kernel_size)
#     stride = torch_utils._pair(stride)
#     padding = torch_utils._pair(padding)
#     dilation = torch_utils._pair(dilation)
#     out_dim_0 = (in_dim[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
#     out_dim_1 = (in_dim[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
#     return out_dim_0, out_dim_1
#
#
# def get_conv_module_output_shape(input_dim, module):
#     dim = input_dim
#     for module_layer in module:
#         if isinstance(module_layer, Conv):
#             for layer in module_layer.layers:
#                 if isinstance(layer, nn.Conv2d):
#                     dim = get_conv2d_layer_output_shape(dim, layer.kernel_size, layer.stride, layer.padding)
#                 elif isinstance(layer, nn.MaxPool2d):
#                     dim = (int(dim[0] / 2),
#                            int(dim[1] / 2))
#     return dim



# 定义通道注意力模块（如SE模块）
# #
# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction_ratio, in_channels)  # 移除 Sigmoid
#         )
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))  # 输出未归一化
#         max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
#         out = avg_out + max_out
#         out = torch.sigmoid(out).unsqueeze(2).unsqueeze(3)  # 仅此处应用 Sigmoid
#         return x * out.expand_as(x)
# # 新增空间注意力模块
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv(x)
#         return self.sigmoid(x)
# # 在残差模块（Residual）中添加通道注意力
# class Residual(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction_ratio=16):
#         super(Residual, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
#         self.bn2 = nn.BatchNorm2d(out_channels // 2)
#         self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(out_channels // 2)
#         self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
#         # 修正跳跃连接：添加BN
#         self.skip = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1),
#             nn.BatchNorm2d(out_channels)
#         ) if in_channels != out_channels else None
#         self.relu = nn.ReLU()
#         self.ca = ChannelAttention(out_channels, reduction_ratio)
#         self.sa = SpatialAttention()
#
#     def forward(self, x):
#         residual = x if self.skip is None else self.skip(x)
#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.conv1(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn3(out)
#         out = self.relu(out)
#         out = self.conv3(out)  # 调整顺序
#         out = self.ca(out)
#         out = out * self.sa(out)
#         out += residual
#         return self.relu(out)  # 可选：根据设计决定是否添加