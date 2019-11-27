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
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.name = 'PATN'

        self.style_block = ResnetBlock(64, 'same', 'instance', use_dropout=True, use_bias=True)

    def set_input(self, inputs):
        self.input = inputs

    def forward(self):
        pass


    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass
