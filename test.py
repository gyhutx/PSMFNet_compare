import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets, Polyp_datasets
from tensorboardX import SummaryWriter
# from models.vmunet.vmunet import VMUNet
from models.vmunet.vmunet_v2 import SkinSTI_Net

from engine_poly import *
import os
import sys

from utils import *
from configs.config_setting_v2_poly import setting_config

import warnings

warnings.filterwarnings("ignore")


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    experiment_name = config.experiment_name  # 从 config 里读，或者自己写

    logger.info(f'===== Starting Experiment: {experiment_name} =====')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    # train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_dataset = Polyp_datasets(config.data_path, config, train=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)

    # val_dataset = NPY_datasets(config.data_path, config, train=False)

    val_loader_dict = {}
    for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        val_dataset = Polyp_datasets(config.data_path, config, train=False, test_dataset=dataset)
        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True)

        val_loader_dict[dataset] = val_loader

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'SkinSTI_Net':
        model = SkinSTI_Net(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
            deep_supervision=model_cfg['deep_supervision'],
        )
        model.logger = logger
        model.load_from()

    else:
        raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1


    # 测试的步骤：也要按照 Polyp 的文件夹列表

    print('#----------Testing----------#')
    best_weight = torch.load('/gpfs/home/WB23301078/run_test/mamba/pymamba/poly/results/best_SkinSTI_Net_polyp_test_g_parameter_d29_m05_y2025_15h_50m_39s/checkpoints/best-epoch138-loss0.6831.pth',
                             map_location=torch.device('cpu'))
    model.load_state_dict(best_weight)
    for name in ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300']:
        val_loader_t = val_loader_dict[name]
        loss = test_one_epoch(
            val_loader_t,
            model,
            criterion,
            logger,
            config,
            test_data_name=name
        )



if __name__ == '__main__':
    config = setting_config
    main(config)