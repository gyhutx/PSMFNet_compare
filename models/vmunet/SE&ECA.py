from .vmamba import VSSM
# from vmamba import VSSM # debug use
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import torch
from torch import nn
from torch.nn import init


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


# 定义ECA注意力模块的类
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


class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1) for _ in range(4)])

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):  # [f1,f2,f3,f4]
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                  mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return ans


class VMUNetV2(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 mid_channel=48,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 deep_supervision=True
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        # SDI
        self.ca_1 = ChannelAttention(2 * mid_channel)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(4 * mid_channel)
        self.sa_2 = SpatialAttention()
        # TODO 320 or mid_channel * 8?
        self.ca_3 = ChannelAttention(8 * mid_channel)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(16 * mid_channel)
        self.sa_4 = SpatialAttention()

        self.Translayer_1 = BasicConv2d(2 * mid_channel, mid_channel, 1)
        self.Translayer_2 = BasicConv2d(4 * mid_channel, mid_channel, 1)
        self.Translayer_3 = BasicConv2d(8 * mid_channel, mid_channel, 1)
        self.Translayer_4 = BasicConv2d(16 * mid_channel, mid_channel, 1)

        # 定义SE模块
        self.se_1 = SEAttention(channel=mid_channel)  # 这个SE模块用来处理f1
        self.se_2 = SEAttention(channel=mid_channel)  # 用来处理f2
        self.se_3 = SEAttention(channel=mid_channel)  # 用来处理f3
        self.se_4 = SEAttention(channel=mid_channel)  # 用来处理f4

        # 定义ECA模块
        self.eca_1 = ECAAttention(kernel_size=3)  # ECA模块
        self.eca_2 = ECAAttention(kernel_size=3)
        self.eca_3 = ECAAttention(kernel_size=3)
        self.eca_4 = ECAAttention(kernel_size=3)

        # 定义SDI模块
        self.sdi_1 = SDI(mid_channel)
        self.sdi_2 = SDI(mid_channel)
        self.sdi_3 = SDI(mid_channel)
        self.sdi_4 = SDI(mid_channel)

        self.seg_outs = nn.ModuleList([
            nn.Conv2d(mid_channel, num_classes, 1, 1) for _ in range(4)])

        self.deconv2 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv4 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv5 = nn.ConvTranspose2d(mid_channel, mid_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv6 = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           )

    def forward(self, x):
        seg_outs = []
        if x.size()[1] == 1:  # 如果是灰度图，就将1个channel 转为3个channel
            x = x.repeat(1, 3, 1, 1)
        f1, f2, f3, f4 = self.vmunet(x)

        # b h w c --> b c h w
        f1 = f1.permute(0, 3, 1, 2)
        f2 = f2.permute(0, 3, 1, 2)
        f3 = f3.permute(0, 3, 1, 2)
        f4 = f4.permute(0, 3, 1, 2)

        # use sdi
        f1 = self.ca_1(f1) * f1
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1)

        f2 = self.ca_2(f2) * f2
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2)

        f3 = self.ca_3(f3) * f3
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3)

        f4 = self.ca_4(f4) * f4
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4)

        f41 = self.sdi_4([f1, f2, f3, f4], f4)  # [2, 48, 8, 8]
        f31 = self.sdi_3([f1, f2, f3, f4], f3)  # [2, 48, 16, 16]
        f21 = self.sdi_2([f1, f2, f3, f4], f2)  # [2, 48, 32, 32]
        f11 = self.sdi_1([f1, f2, f3, f4], f1)  # [2, 48, 64, 64]

        # 在SDI模块之后应用SE模块
        f41 = self.se_4(f41)  # 将SE模块应用于f41
        f31 = self.se_3(f31)  # 将SE模块应用于f31
        f21 = self.se_2(f21)  # 将SE模块应用于f21
        f11 = self.se_1(f11)  # 将SE模块应用于f11

        # 应用ECA模块
        f41 = self.eca_4(f41)
        f31 = self.eca_3(f31)
        f21 = self.eca_2(f21)
        f11 = self.eca_1(f11)

        # 函数seg_outs 输出列表也是 seg_outs 只是名字相同
        seg_outs.append(self.seg_outs[0](f41))  # seg_outs[0] [2, 1, 8, 8]

        y = self.deconv2(f41) + f31
        seg_outs.append(self.seg_outs[1](y))  # seg_outs[1] [2, 1, 16, 16]

        y = self.deconv3(y) + f21
        seg_outs.append(self.seg_outs[2](y))  # seg_outs[2] [2, 1, 32, 32]

        y = self.deconv4(y) + f11
        seg_outs.append(self.seg_outs[3](y))  # seg_outs[3] [2, 1, 64, 64]

        for i, o in enumerate(seg_outs):  # 4 倍上采样
            seg_outs[i] = F.interpolate(o, scale_factor=4, mode='bilinear')

        if self.deep_supervision:

            temp = seg_outs[::-1]  # 0 [2, 1, 256, 256] 1 [2, 1, 128, 128]
            out_0 = temp[0]
            out_1 = temp[1]
            out_1 = self.deconv6(out_1)
            return torch.sigmoid(out_0 + out_1)  # [2, 1, 256, 256]
        else:
            if self.num_classes == 1:
                return torch.sigmoid(seg_outs[-1])
            else:
                return seg_outs[-1]

    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict),
                                                                                       len(pretrained_dict),
                                                                                       len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            # model_dict = self.vmunet.state_dict()
            # modelCheckpoint = torch.load(self.load_ckpt_path)
            # # 下面 是 layers up
            # pretrained_odict = modelCheckpoint['model']
            # pretrained_dict = {}
            # for k, v in pretrained_odict.items():
            #     if 'layers.0' in k:
            #         new_k = k.replace('layers.0', 'layers_up.3')
            #         pretrained_dict[new_k] = v
            #     elif 'layers.1' in k:
            #         new_k = k.replace('layers.1', 'layers_up.2')
            #         pretrained_dict[new_k] = v
            #     elif 'layers.2' in k:
            #         new_k = k.replace('layers.2', 'layers_up.1')
            #         pretrained_dict[new_k] = v
            #     elif 'layers.3' in k:
            #         new_k = k.replace('layers.3', 'layers_up.0')
            #         pretrained_dict[new_k] = v
            # # 过滤操作
            # new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            # model_dict.update(new_dict)
            # # 打印出来，更新了多少的参数
            # print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            # self.vmunet.load_state_dict(model_dict)

            # # 找到没有加载的键(keys)
            # not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            # print('Not loaded keys:', not_loaded_keys)
            # print("decoder loaded finished!")


if __name__ == '__main__':
    pretrained_path = '/raid/code/mamba_all/VM-UNet/pre_trained_weights/vmamba_small_e238_ema.pth'
    model = VMUNetV2(load_ckpt_path=pretrained_path, deep_supervision=True).cuda()
    model.load_from()
    x = torch.randn(2, 3, 256, 256).cuda()
    predict = model(x)
    # print(predict.shape)  #  deep_supervision true   predict[0] [2, 1, 256, 256] , predict[1] [2, 1, 128, 128] 这两项用于监督



