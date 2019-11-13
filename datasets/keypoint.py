# -*- coding: utf-8 -*-
"""
@Project:   pytorch-train
@File   :   keypoint
@Author :   TonyMao@AILab
@Date   :   2019/11/12
@Desc   :   None
"""
from datasets.base import BaseDataset
from datasets.utils import get_transform


class KeypointDataset(BaseDataset):
    def __init__(self, opt):
        super(KeypointDataset, self).__init__()
        self.opt = opt
        self.preprocess = get_transform(opt)
        self.instance_list = []

    def __len__(self):
        return len(self.instance_list)




if __name__ == '__main__':
    KeypointDataset(opt=None)
