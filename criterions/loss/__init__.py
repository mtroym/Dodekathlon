import torch
import torch.nn as nn

from .DefGan_loss import NNLoss, IOULoss
from .adain_loss import style_loss_dict, content_loss, style_loss

loss_dict = {
    "MSE": nn.MSELoss(),
    "BCE": nn.BCELoss(),
    "L1": nn.L1Loss(),
    "NNL": NNLoss(),
    "IOU": IOULoss(),
    "WST": lambda pred_r, pred_f: - torch.mean(pred_r) + torch.mean(pred_f),  # wasserstein loss
    "adain_content": content_loss,
    "adain_style_dict": style_loss_dict,
    "adain_style": style_loss
}


def create_loss_single(lamda, loss):
    return lambda *inputs: lamda * loss_dict[loss](*inputs)


def create_loss(opt):
    train_loss = opt.loss
    print(train_loss.items())
    loss = {}
    for k, v in train_loss.items():
        loss[k] = create_loss_single(*v)
    print('-> creating loss...')
    return loss
