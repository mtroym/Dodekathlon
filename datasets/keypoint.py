# -*- coding: utf-8 -*-
"""
@Project:   pytorch-train
@File   :   keypoint
@Author :   TonyMao@AILab
@Date   :   2019/11/12
@Desc   :   None
"""
import os

import numpy as np
import torch
import yaml
from PIL import Image

from datasets.utils import get_transform


class KeypointDataset:
    def __init__(self, opt, split="train"):
        super(KeypointDataset, self).__init__()
        self.base_dir = None
        self.split = split
        self.opt = opt
        self.data_info_path = os.path.join(self.opt.data_gen, self.opt.dataset + "_{}_info.npy".format(split))
        self.preprocess = get_transform(opt)
        self.configure = yaml.load(open(os.path.join(self.opt.configure_path, self.opt.dataset + '.yaml'), 'r'))

        if os.path.exists(self.data_info_path) and not self.opt.refresh:
            print("=> load dataset information from {}".format(self.data_info_path))
            self.data_info = torch.load(self.data_info_path)
        else:
            print("=> create dataset information....")
            self.data_info = {
                "base_dir": os.path.abspath(os.path.join(self.configure["base_dir"], self.configure["{}_path".format(split)])),
                "blob"    : self._get_blob()
            }
            with open(os.path.join(self.configure["base_dir"], self.configure["train_pair_path"]), "r") as f:
                pairs = f.read().split("\n")[1:]
                self.pairs = list(map(lambda x: x.split(','), pairs))
            self.data_info["pairs"] = self.pairs
            self.data_info["total_num"] = len(self.data_info["pairs"])
            torch.save(self.data_info, self.data_info_path)
            print("=> saved dataset information from {}".format(self.data_info_path))
        for k, v in self.data_info.items():
            self.__setattr__(k, v)

    def _mapping_image2kp(self, image_path):
        return os.path.join(self.configure["base_dir"],
                            self.configure["train_keypoint_path"],
                            image_path.split("/")[-1] + ".npy")

    @staticmethod
    def _get_blob(src=None, kp0=None, trg=None, kp1=None):
        return {"Source": src, "Source_KP": kp0, "Target": trg, "Target_KP": kp1}

    def __len__(self):
        return len(self.pairs)

    def _get_one(self, name):
        img_path = os.path.join(self.base_dir, name)
        kp_path = self._mapping_image2kp(image_path=img_path)
        image = Image.open(img_path, 'r')
        assert image is not None, "{}, image not exists, img_path".format(img_path)
        assert os.path.exists(kp_path), "{}, kp file not exists".format(kp_path)

        keypoint = np.load(kp_path).transpose([-1, 0, 1])
        return image, keypoint

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        (src, src_kp), (trg, trg_kp) = self._get_one(os.path.join(self.base_dir, pair[0])), \
                                       self._get_one(os.path.join(self.base_dir, pair[1]))
        src = self.preprocess(src)
        trg = self.preprocess(trg)
        src_kp = torch.from_numpy(src_kp)
        trg_kp = torch.from_numpy(trg_kp)

        blob = self._get_blob(src, src_kp, trg, trg_kp)
        return blob


if __name__ == '__main__':
    pass
    # data = KeypointDataset(opt=None)
    # print(len(data[1]))
