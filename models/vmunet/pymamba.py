
from .vmamba import VSSM
from .MYmodule import *
# from vmamba import VSSM # debug use
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import torch
from torch import nn
from torch.nn import init

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

        # CA
        self.ca = ChannelAttention(2 * mid_channel)

        # SA
        self.sa = SpatialAttention()


        self.Translayer_1 = BasicConv2d(2 * mid_channel, mid_channel, 1)
        self.Translayer_2 = BasicConv2d(4 * mid_channel, mid_channel, 1)
        self.Translayer_3 = BasicConv2d(8 * mid_channel, mid_channel, 1)
        self.Translayer_4 = BasicConv2d(16 * mid_channel, mid_channel, 1)

        # 定义CFM模块
        self.CFM = CFM(mid_channel)

        # 定义采样输出
        self.out_CFM = nn.Conv2d(mid_channel, 1, 1)
        self.out_SAM = nn.Conv2d(mid_channel, 1, 1)

        # 定义SAM模块
        self.SAM = SAM()

        # 定义下采样
        self.downhalf = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

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

        # decoder:
        # use CFM
        f2_1 = self.Translayer_2(f2)
        f3_1 = self.Translayer_3(f3)
        f4_1 = self.Translayer_4(f4)
        cfm_feature = self.CFM(f4_1, f3_1, f2_1)

        T1 = self.out_CFM(cfm_feature)

        # use CIM
        f1 = self.ca(f1) * f1  # channel attention
        cim_feature = self.sa(f1) * f1  # spatial attention

        # use SAM
        T2 = self.Translayer_1(cim_feature)
        T2 = self.downhalf(T2)
        sam_feature = self.SAM(cfm_feature, T2)

        # 两个预测输出
        prediction1 = self.out_CFM(cfm_feature)
        prediction2 = self.out_SAM(sam_feature)

        prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear')
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')
        return prediction1_8, prediction2_8

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
    pretrained_path = '/gpfs/home/WB23301078/mamba/vmunet/VM-UNet-main/pre_trained_weights/vmamba_small_e238_ema.pth'
    model = VMUNetV2(load_ckpt_path=pretrained_path, deep_supervision=True).cuda()

    # 打印模型结构树
    print('模型结构：')
    print(model)

    # 打印所有参数key
    print('\n所有参数key（state_dict的key）：')
    for key in model.state_dict().keys():
        print(key)

    # 打印所有层名字和类型
    print('\n所有层名字和类型：')
    for name, module in model.named_modules():
        print(name, '→', module.__class__.__name__)

    # 前向测试
    x = torch.randn(2, 3, 256, 256).cuda()
    predict1, predict2 = model(x)
    print(f'predict1: {predict1.shape}')
    print(f'predict2: {predict2.shape}')




