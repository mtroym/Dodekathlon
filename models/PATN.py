# -*- coding: utf-8 -*-
"""
@Project:   pytorch-train
@File   :   PATN
@Author :   TonyMao@AILab
@Date   :   2019/11/26
@Desc   :   None
"""
import os

import torch
import torch.nn as nn

from .blocks import ResnetBlock


class PATN(nn.Module):
    def __init__(self, opt):
        super(PATN, self).__init__()
        self.name = 'PATN'
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = True
        self.dtype = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        self.style_block = ResnetBlock(64, 'same', 'instance', use_dropout=True, use_bias=True)

    def set_input(self, inputs):
        self.input = inputs

    def forward(self):
        pass


class PATN_Discriminator(nn.Module):
    def __init__(self):
        super(PATN_Discriminator, self).__init__()

