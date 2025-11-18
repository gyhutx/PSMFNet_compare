
from .vmamba import VSSM
from .MYmodule import *
# from vmamba import VSSM # debug use
import torch
from torch import nn
import torch.nn.functional as F


class SkinSTI_Net(nn.Module):
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

        # log
        self.logger = None
        self.tflog = True

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

        # 定义CTI交互模块
        self.sti = SimpleCTI(in_channels=mid_channel)

        # 定义下采样
        self.downhalf = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           )

    def forward(self, x):
        if self.tflog:
            self.logger.info("\n \t Forward pass started.")

        if x.size()[1] == 1:  # 如果是灰度图，就将1个channel 转为3个channel
            x = x.repeat(1, 3, 1, 1)

        if self.tflog:
            self.logger.info("Passing input through vmunet encoder.")
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

        if self.tflog:
            self.logger.info("Applying CFM module.")
        cfm_feature = self.CFM(f4_1, f3_1, f2_1)

        T1 = cfm_feature

        # use CIM
        if self.tflog:
            self.logger.info("Applying CA and spatial attention (SA) for CIM.")
        f1 = self.ca(f1) * f1  # channel attention
        cim_feature = self.sa(f1) * f1  # spatial attention

        #use ECAM
        # if self.tflog:
        #     self.logger.info("Applying ECA and ESA for ECAM.")
        # cim_feature = self.ecam(f1)

        # use STI

        f1_1 = self.Translayer_1(cim_feature)

        if self.tflog:
            self.logger.info("Applying STI module with cfm_feature and cim_feature.")
        sti_feature = self.sti(cfm_feature, f1_1)

        # use SAM
        T2 = f1_1

        T2 = self.downhalf(T2)

        if self.tflog:
            self.logger.info("not Applying SAM module.")
        # sam_feature = self.SAM(sti_feature, T2)

        # 两个预测输出
        if self.tflog:
            self.logger.info("Generating prediction1 and prediction2.")
        prediction1 = self.out_CFM(cfm_feature)
        prediction2 = self.out_SAM(sti_feature)

        if self.tflog:
            self.logger.info("Upsampling predictions by scale factor 8. \n"
                             "\t Use S1 for main loss, T1 for aux loss, loss function use original loss in isic_all.")
        prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear')
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')

        if self.tflog:
            self.logger.info("Forward pass completed.")
            self.tflog = False
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



if __name__ == '__main__':
    import logging
    import logging.handlers

    def get_logger(name, log_dir):
        '''
        Args:
            name(str): name of logger
            log_dir(str): path of log
        '''

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        info_name = os.path.join(log_dir, '{}.info.log'.format(name))
        info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                                 when='D',
                                                                 encoding='utf-8')
        info_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        info_handler.setFormatter(formatter)

        logger.addHandler(info_handler)

        return logger

    pretrained_path = '/gpfs/home/WB23301078/mamba/vmunet/VM-UNet-main/pre_trained_weights/vmamba_small_e238_ema.pth'
    model = SkinSTI_Net(load_ckpt_path=pretrained_path, deep_supervision=True).cuda()
    model.logger = get_logger('model_test', './')
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
    x1 = torch.randn(2, 3, 256, 256).cuda()
    x2 = torch.randn(2, 3, 256, 256).cuda()
    predict11, predict12 = model(x1)
    predict21, predict22= model(x2)
    print(f'predict1: {predict11.shape}')
    print(f'predict2: {predict22.shape}')



















