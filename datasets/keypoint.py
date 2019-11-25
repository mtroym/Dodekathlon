# -*- coding: utf-8 -*-
"""
@Project:   pytorch-train
@File   :   keypoint
@Author :   TonyMao@AILab
@Date   :   2019/11/12
@Desc   :   None
"""
import csv
import os

import torch
import yaml

from datasets.utils import get_transform


class KeypointDataset:
    def __init__(self, opt, split="train"):
        super(KeypointDataset, self).__init__()
        self.keypoint_path = None
        self.instance_list = None
        self.split = split
        self.opt = opt
        self.data_info_path = os.path.join(self.opt.data_gen, self.opt.dataset + "_{}_info.npy".format(split))
        self.preprocess = get_transform(opt)
        self.configure = yaml.load(open(os.path.join(self.opt.configure_path, self.opt.dataset + '.yaml'), 'r'))
        with open(self.configure["train_pair_path"], "r") as f:
            pairs = f.read().split("\n")[1:]

        if os.path.exists(self.data_info_path) and self.opt.refresh:
            print("=> load dataset information from {}".format(self.data_info_path))
            self.data_info = torch.load(self.data_info_path)
        else:
            self.pairs = list(map(lambda x: x.split(','), pairs))
            self.data_info = {
                "image_path" : os.path.abspath(self.opt.__getattribute__("{}_path".format(split))),
                "target_path": os.path.abspath(self.opt.__getattribute__("{}_keypoint_path".format(split))),
                "blob"       : {"image": None, "KP": None, "target": None, "target_KP": None}
            }
            self.data_info["instance_list"] = os.listdir(self.data_info["image_path"])
            self.data_info["total_num"] = len(self.data_info["instance_list"])
            torch.save(self.data_info, self.data_info_path)
        for k, v in self.data_info.items():
            self.__setattr__(k, v)

    @staticmethod
    def _mapping_image2kp(self, image_path):
        return image_path.replace("{}".format(self.split), "{}K".format(self.split))

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):
        source_data = self.instance_list[idx]
        source_kp = self.keypoint_path[idx]


if __name__ == '__main__':
    KeypointDataset(opt=None)
