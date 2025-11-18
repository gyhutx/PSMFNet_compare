import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# 定义 Deformable 实现的 STI

# class DeformableCrossAttention(nn.Module):
#     """
#     可变形跨注意力模块，基于Deformable DETR的实现，单尺度版本
#     """
#     def __init__(self, embed_dim, num_heads=4, num_points=4, num_levels=1):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.num_points = num_points
#         self.num_levels = num_levels
#
#         # 确保 embed_dim 可被 num_heads 整除
#         assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
#
#         # 查询、键、值投影
#         self.query_proj = nn.Linear(embed_dim, embed_dim)
#         self.key_value_proj = nn.Linear(embed_dim, embed_dim * 2)
#         # 每头投影
#         self.head_proj = nn.Linear(embed_dim, embed_dim // num_heads)
#         # 偏移量预测
#         self.offset_conv = nn.Conv2d(embed_dim, 2 * num_levels * num_heads * num_points, kernel_size=1)
#         # 注意力权重预测
#         self.attn_weight = nn.Linear(embed_dim, num_heads * num_points)
#         # 输出投影
#         self.output_proj = nn.Linear(embed_dim, embed_dim)
#         self.norm = nn.LayerNorm(embed_dim)
#
#     def forward(self, queries, feature_map):
#         """
#         queries: [B, N, C]
#         feature_map: [B, C, H, W]
#         """
#         B, N, C = queries.shape
#         _, _, H, W = feature_map.shape
#
#         # 投影查询
#         q = self.query_proj(queries)  # [B, N, C]
#         # 投影键和值（仅用于键，值直接从 feature_map 采样）
#         kv = self.key_value_proj(feature_map.flatten(2).transpose(1, 2))  # [B, H*W, 2C]
#         k, _ = kv.chunk(2, dim=-1)  # [B, H*W, C]
#
#         # 预测偏移量
#         offsets = self.offset_conv(feature_map)  # [B, 2*num_levels*num_heads*num_points, H, W]
#         offsets = offsets.view(B, self.num_levels, self.num_heads, self.num_points, 2, H, W)
#         offsets = offsets.permute(0, 2, 1, 3, 5, 6, 4)  # [B, num_heads, num_levels, num_points, H, W, 2]
#
#         # 生成参考点网格
#         grid_y, grid_x = torch.meshgrid(torch.linspace(0, H-1, H, device=feature_map.device),
#                                         torch.linspace(0, W-1, W, device=feature_map.device), indexing='ij')
#         reference_points = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
#         reference_points = reference_points.view(1, 1, 1, 1, H, W, 2)  # [1, 1, 1, 1, H, W, 2]
#
#         # 计算采样点
#         sampling_locations = reference_points + offsets  # [B, num_heads, num_levels, num_points, H, W, 2]
#         sampling_locations = sampling_locations.view(B, self.num_heads, self.num_levels * self.num_points, H, W, 2)
#
#         # 归一化采样点到[-1, 1]
#         sampling_locations[..., 0] = sampling_locations[..., 0] / (W - 1) * 2 - 1
#         sampling_locations[..., 1] = sampling_locations[..., 1] / (H - 1) * 2 - 1
#
#         # 采样特征
#         sampling_locations = sampling_locations.reshape(B * self.num_heads, self.num_levels * self.num_points, H * W, 2)
#         sampled_feats = F.grid_sample(feature_map.repeat(self.num_heads, 1, 1, 1),
#                                       sampling_locations,
#                                       align_corners=False)  # [B*num_heads, C, num_points, H*W]
#         sampled_feats = sampled_feats.view(B, self.num_heads, C, self.num_points, H * W)
#         sampled_feats = sampled_feats.permute(0, 1, 4, 3, 2)  # [B, num_heads, H*W, num_points, C]
#         sampled_feats = self.head_proj(sampled_feats)  # [B, num_heads, H*W, num_points, C//num_heads]
#
#         # 计算注意力权重
#         attn_weights = self.attn_weight(queries)  # [B, N, num_heads*num_points]
#         attn_weights = attn_weights.view(B, N, self.num_heads, self.num_points)
#         attn_weights = F.softmax(attn_weights, dim=-1)  # [B, N, num_heads, num_points]
#         attn_weights = attn_weights.permute(0, 2, 1, 3)  # [B, num_heads, N, num_points]
#
#         # 聚合特征
#         out = (attn_weights.unsqueeze(-1) * sampled_feats).sum(dim=3)  # [B, num_heads, H*W, C//num_heads]
#         out = out.permute(0, 2, 1, 3).reshape(B, N, self.num_heads * (C // self.num_heads))
#         out = self.output_proj(out)
#         out = self.norm(out)
#         return out
#
# class SimpleCTI(nn.Module):
#     """
#     简化版 Cross-scale Token Interaction with Deformable Attention
#     输入两个特征图，通过可变形跨注意力进行融合，降低计算复杂度
#     """
#     def __init__(self, in_channels, attn_heads=4, num_points=4, use_cnn=True):
#         super().__init__()
#         self.use_cnn = use_cnn
#         assert in_channels % attn_heads == 0, "in_channels must be divisible by attn_heads"
#         self.deform_attn = DeformableCrossAttention(
#             embed_dim=in_channels,
#             num_heads=attn_heads,
#             num_points=num_points,
#             num_levels=1  # 单尺度
#         )
#         self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, 1)
#         if use_cnn:
#             self.cnn_block = nn.Sequential(
#                 nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(in_channels, in_channels, 1),
#                 nn.BatchNorm2d(in_channels),
#                 nn.ReLU(inplace=True),
#             )
#
#     def forward(self, x_main, x_aux):
#         """
#         x_main: [B, C, H, W]
#         x_aux:  [B, C, H, W] 或 [B, C, H', W']
#         """
#         B, C, H, W = x_main.shape
#         if x_aux.shape[2:] != (H, W):
#             x_aux = F.interpolate(x_aux, size=(H, W), mode='bilinear', align_corners=False)
#
#         # 展平为token序列
#         x_main_flat = x_main.flatten(2).transpose(1, 2)  # [B, N, C]
#         x_aux_flat = x_aux.flatten(2).transpose(1, 2)    # [B, N, C]
#
#         # 双向可变形跨注意力
#         out_main = self.deform_attn(queries=x_main_flat, feature_map=x_aux)
#         out_aux = self.deform_attn(queries=x_aux_flat, feature_map=x_main)
#
#         # 重塑为特征图
#         out_main = out_main.transpose(1, 2).view(B, C, H, W)
#         out_aux = out_aux.transpose(1, 2).view(B, C, H, W)
#
#         # 通道拼接与融合
#         fused = torch.cat([out_main, out_aux], dim=1)  # [B, 2C, H, W]
#         fused = self.fusion_conv(fused)                # [B, C, H, W]
#
#         # 可选CNN增强
#         if self.use_cnn:
#             fused = self.cnn_block(fused)
#
#         return fused


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化层，将空间维度压缩为1x1
        # 定义一个1D卷积，用于处理通道间的关系，核大小可调，padding保证输出通道数不变
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数，用于激活最终的注意力权重

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')  # 对Conv2d层使用Kaiming初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)  # 批归一化层权重初始化为1
                init.constant_(m.bias, 0)  # 批归一化层偏置初始化为0
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)  # 全连接层权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 全连接层偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        y = self.gap(x)  # 对输入x应用全局平均池化，得到bs,c,1,1维度的输出
        y = y.squeeze(-1).permute(0, 2, 1)  # 移除最后一个维度并转置，为1D卷积准备，变为bs,1,c
        y = self.conv(y)  # 对转置后的y应用1D卷积，得到bs,1,c维度的输出
        y = self.sigmoid(y)  # 应用Sigmoid函数激活，得到最终的注意力权重
        y = y.permute(0, 2, 1).unsqueeze(-1)  # 再次转置并增加一个维度，以匹配原始输入x的维度
        return x * y.expand_as(x)  # 将注意力权重应用到原始输入x上，通过广播机制扩展维度并执行逐元素乘法

class SimpleCTI(nn.Module):
    """
    简化版 Cross-scale Token Interaction
    输入两个不同分辨率特征，自动flatten，做token级融合与交互。
    """

    def __init__(self, in_channels, attn_heads=4, fusion_hidden=None, use_cnn=True):
        super().__init__()
        self.use_cnn = use_cnn
        self.attn_heads = attn_heads
        hidden = fusion_hidden or in_channels

        # Token交互: 多头自注意力
        self.token_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=attn_heads, batch_first=True)

        # 通道融合: 1x1卷积
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, 1)

        if use_cnn:
            # 可选：CNN局部增强
            self.cnn_block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x_main, x_aux):
        """
        x_main: [B, C, H, W]
        x_aux:  [B, C, H, W] 或 [B, C, H', W']
        """
        B, C, H, W = x_main.shape

        # 若x_aux与x_main分辨率不同，resize到一致
        if x_aux.shape[2:] != (H, W):
            x_aux = F.interpolate(x_aux, size=(H, W), mode='bilinear', align_corners=False)

        # flatten到token序列
        x1_token = x_main.flatten(2).transpose(1, 2)  # [B, N, C]
        x2_token = x_aux.flatten(2).transpose(1, 2)   # [B, N, C]

        # 拼接两个token序列，做self-attention交互
        tokens = torch.cat([x1_token, x2_token], dim=1)  # [B, 2N, C]
        attn_out, _ = self.token_attn(tokens, tokens, tokens)  # [B, 2N, C]

        # 拆分还原
        out1 = attn_out[:, :H*W, :].transpose(1, 2).view(B, C, H, W)
        out2 = attn_out[:, H*W:, :].transpose(1, 2).view(B, C, H, W)

        # 拼接+通道融合
        fused = torch.cat([out1, out2], dim=1)  # [B, 2C, H, W]
        fused = self.fusion_conv(fused)         # [B, C, H, W]

        # 可选：CNN增强
        if self.use_cnn:
            fused = self.cnn_block(fused)

        return fused

class SEAttention(nn.Module):
    # 初始化SE模块，channel为通道数，reduction为降维比率
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层，将特征图的空间维度压缩为1x1
        self.fc = nn.Sequential(  # 定义两个全连接层作为激励操作，通过降维和升维调整通道重要性
            nn.Linear(channel, channel // reduction, bias=False),  # 降维，减少参数数量和计算量
            nn.ReLU(inplace=True),  # ReLU激活函数，引入非线性
            nn.Linear(channel // reduction, channel, bias=False),  # 升维，恢复到原始通道数
            nn.Sigmoid()  # Sigmoid激活函数，输出每个通道的重要性系数
        )

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():  # 遍历模块中的所有子模块
            if isinstance(m, nn.Conv2d):  # 对于卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用Kaiming初始化方法初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):  # 对于批归一化层
                init.constant_(m.weight, 1)  # 权重初始化为1
                init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):  # 对于全连接层
                init.normal_(m.weight, std=0.001)  # 权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入x的批量大小b和通道数c
        y = self.avg_pool(x).view(b, c)  # 通过自适应平均池化层后，调整形状以匹配全连接层的输入
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层计算通道重要性，调整形状以匹配原始特征图的形状
        return x * y.expand_as(x)  # 将通道重要性系数应用到原始特征图上，进行特征重新校准


class MRFPv2(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, act_layer=nn.GELU):
        super(MRFPv2, self).__init__()
        hidden_channels = hidden_channels or in_channels
        half = hidden_channels // 2

        # 上采样分支（标准卷积）
        self.up_conv3 = nn.Conv2d(half, half, 3, padding=1, groups=1)
        self.up_conv5 = nn.Conv2d(half, half, 5, padding=2, groups=1)
        self.up_bn = nn.BatchNorm2d(hidden_channels)
        self.up_act = act_layer()

        # 原始分支（标准卷积）
        self.mid_conv3 = nn.Conv2d(half, half, 3, padding=1, groups=1)
        self.mid_conv5 = nn.Conv2d(half, half, 5, padding=2, groups=1)
        self.mid_bn = nn.BatchNorm2d(hidden_channels)
        self.mid_act = act_layer()

        # 下采样分支（标准卷积）
        self.down_conv3 = nn.Conv2d(half, half, 3, padding=1, groups=1)
        self.down_conv5 = nn.Conv2d(half, half, 5, padding=2, groups=1)
        self.down_bn = nn.BatchNorm2d(hidden_channels)
        self.down_act = act_layer()

        # 融合输出
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            act_layer()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        half = C // 2

        # ----------------- 上采样分支 -----------------
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x_up1, x_up2 = x_up[:, :half], x_up[:, half:]
        x_up3 = torch.cat([self.up_conv3(x_up1), self.up_conv5(x_up2)], dim=1)
        x_up3 = self.up_act(self.up_bn(x_up3))
        x_up3 = F.adaptive_avg_pool2d(x_up3, (H, W))

        # ----------------- 原始分支 -------------------
        x_mid1, x_mid2 = x[:, :half], x[:, half:]
        x_mid = torch.cat([self.mid_conv3(x_mid1), self.mid_conv5(x_mid2)], dim=1)
        x_mid = self.mid_act(self.mid_bn(x_mid))

        # ----------------- 下采样分支 -----------------
        x_down = F.adaptive_avg_pool2d(x, (H // 2, W // 2))
        x_down = F.interpolate(x_down, size=(H, W), mode='bilinear', align_corners=False)
        x_down1, x_down2 = x_down[:, :half], x_down[:, half:]
        x_down3 = torch.cat([self.down_conv3(x_down1), self.down_conv5(x_down2)], dim=1)
        x_down3 = self.down_act(self.down_bn(x_down3))

        # ----------------- 融合输出 -------------------
        x_cat = torch.cat([x_up3, x_mid, x_down3], dim=1)
        x_out = self.fuse(x_cat)

        return x_out


class MRFP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MultiDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class MultiDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        dim1 = dim
        dim = dim // 2

        self.dwconv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

        self.dwconv3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv4 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

        self.dwconv5 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv6 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(dim1)

        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(dim1)

        self.act3 = nn.GELU()
        self.bn3 = nn.BatchNorm2d(dim1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()

        x11, x12 = x1[:, :C // 2, :, :], x1[:, C // 2:, :, :]
        x11 = self.dwconv1(x11)  # BxCxHxW
        x12 = self.dwconv2(x12)
        x1 = torch.cat([x11, x12], dim=1)
        x1 = self.act1(self.bn1(x1)).flatten(2).transpose(1, 2)

        x21, x22 = x2[:, :C // 2, :, :], x2[:, C // 2:, :, :]
        x21 = self.dwconv3(x21)
        x22 = self.dwconv4(x22)
        x2 = torch.cat([x21, x22], dim=1)
        x2 = self.act2(self.bn2(x2)).flatten(2).transpose(1, 2)

        x31, x32 = x3[:, :C // 2, :, :], x3[:, C // 2:, :, :]
        x31 = self.dwconv5(x31)
        x32 = self.dwconv6(x32)
        x3 = torch.cat([x31, x32], dim=1)
        x3 = self.act3(self.bn3(x3)).flatten(2).transpose(1, 2)

        x = torch.cat([x1, x2, x3], dim=1)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class SAM(nn.Module):
    def __init__(self, num_in=48, plane_mid=16, mids=4, normalize=False):
        super(SAM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor1 = self.priors(x_mask)
        x_anchor2 = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + (self.conv_extend(x_state))

        return out