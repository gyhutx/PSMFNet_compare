import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn.functional as F
import os

from sklearn.metrics import confusion_matrix
from utils import save_imgs

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def multi_output_structure_loss(outputs, targets):
    if isinstance(outputs, tuple):
        p1, p2 = outputs
        loss1 = structure_loss(p1, targets)
        loss2 = structure_loss(p2, targets)
        return loss1 + loss2
    else:
        return structure_loss(outputs, targets)

def multi_output_structure_loss_alpha(outputs, targets, alpha=0.8):
    if isinstance(outputs, tuple):
        p1, p2 = outputs
        loss1 = structure_loss(p1, targets)
        loss2 = structure_loss(p2, targets)
        return alpha * loss2 + (1 - alpha) * loss1
    else:
        return structure_loss(outputs, targets)


def dynamic_loss(outputs, targets, epoch, total_epochs):
    p1, p2 = outputs
    # 动态调整损失权重
    alpha = max(0.1, (total_epochs - epoch) / total_epochs)
    loss1 = structure_loss(p1, targets)
    loss2 = structure_loss(p2, targets)
    return alpha * loss1 + (1 - alpha) * loss2

def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images)
        loss = multi_output_structure_loss(out, targets)
        # loss = dynamic_loss(out, targets, epoch, 300)
        # loss = multi_output_structure_loss_alpha(out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config,
                    val_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
                   
            # loss = multi_output_structure_loss(out, msk)
            loss = dynamic_loss(out, msk, epoch, 300)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if isinstance(out, tuple):
                out = out[0] + out[1]  # 融合预测

            out = F.interpolate(out, size=msk.shape[2:], mode='bilinear', align_corners=True)
            out = torch.sigmoid(out)
            out = out.squeeze(1).cpu().detach().numpy()

            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if val_data_name is not None:
            log_info = f'val_datasets_name: {val_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f' val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f' val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    # return np.mean(loss_list)
    return miou

# def test_one_epoch(test_loader,
#                     model,
#                     criterion,
#                     logger,
#                     config,
#                     test_data_name=None):
#     # switch to evaluate mode
#     model.eval()
#     preds = []
#     gts = []
#     loss_list = []
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(test_loader)):
#             img, msk = data
#             img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
#
#             out = model(img)
#             loss = multi_output_structure_loss(out, msk)
#
#             loss_list.append(loss.item())
#             msk = msk.squeeze(1).cpu().detach().numpy()
#             gts.append(msk)
#             if type(out) is tuple:
#                 out = out[0] + out[1]
#             out = out.squeeze(1).cpu().detach().numpy()
#             preds.append(out)
#             if i % config.save_interval == 0:
#                 save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)
#
#         preds = np.array(preds).reshape(-1)
#         gts = np.array(gts).reshape(-1)
#
#         y_pre = np.where(preds>=config.threshold, 1, 0)
#         y_true = np.where(gts>=0.5, 1, 0)
#
#         confusion = confusion_matrix(y_true, y_pre)
#         TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]
#
#         accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
#         sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
#         specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
#         f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
#         miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
#
#         if test_data_name is not None:
#             log_info = f'test_datasets_name: {test_data_name}'
#             print(log_info)
#             logger.info(log_info)
#         log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
#                 specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
#         print(log_info)
#         logger.info(log_info)
#
#     # return np.mean(loss_list)
#     return miou

def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    # Create directories for saving outputs
    os.makedirs(config.work_dir + 'outputs', exist_ok=True)
    os.makedirs(config.work_dir + 'outputs/out0', exist_ok=True)
    os.makedirs(config.work_dir + 'outputs/out1', exist_ok=True)
    os.makedirs(config.work_dir + 'outputs/sum', exist_ok=True)

    # Switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            model_out = model(img)
            loss = multi_output_structure_loss(model_out, msk)
            loss_list.append(loss.item())
            msk_np = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk_np)

            if type(model_out) is tuple:
                if i % config.save_interval == 0:
                    for idx in range(2):
                        o = model_out[idx].squeeze(1).cpu().detach().numpy()
                        save_dir = config.work_dir + f'outputs/out{idx}/'
                        save_imgs(img, msk, o, i, save_dir, config.datasets, config.threshold, test_data_name=test_data_name)
                out = (model_out[0] + model_out[1]).squeeze(1).cpu().detach().numpy()
            else:
                out = model_out.squeeze(1).cpu().detach().numpy()

            preds.append(out)
            if i % config.save_interval == 0:
                if type(model_out) is tuple:
                    save_dir = config.work_dir + 'outputs/sum/'
                else:
                    save_dir = config.work_dir + 'outputs/'
                save_imgs(img, msk, out, i, save_dir, config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return miou