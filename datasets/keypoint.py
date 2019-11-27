# -*- coding: utf-8 -*-
"""
@Project:   pytorch-train
@File   :   keypoint
@Author :   TonyMao@AILab
@Date   :   2019/11/12
@Desc   :   None
"""
import os

import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image

from datasets.utils import get_transform


# Example dataset.
class KeypointDataset:
    def __init__(self, opt, split="train"):
        super(KeypointDataset, self).__init__()
        self.base_dir = None
        self.split = split
        self.opt = opt
        self.data_info_path = os.path.join(self.opt.data_gen, self.opt.dataset + "_{}_info.pth.tar".format(split))
        self.num_kp = 18
        self.configure = yaml.load(open(os.path.join(self.opt.configure_path, self.opt.dataset + '.yaml'), 'r'))

        # custom this from multiple datasets.
        if self.opt.dataset == "deepfashion256":
            self.h, self.w = 256, 256
            self.transforms = transforms.Pad((40, 0, 40, 0), padding_mode='edge')
        elif self.opt.dataset == "deepfashion512":
            self.h, self.w = 512, 512
            self.transforms = transforms.Pad((80, 0, 80, 0), padding_mode='edge')

        self.preprocess = transforms.Compose([
            self.transforms,  # custom preprocessing.
            get_transform(opt),
        ])
        if os.path.exists(self.data_info_path) and not self.opt.refresh:
            print("=> load dataset information from {}".format(self.data_info_path))
            self.data_info = torch.load(self.data_info_path)
        else:
            print("=> create dataset information....")
            self.data_info = {
                "base_dir": os.path.abspath(os.path.join(self.configure["base_dir"], self.configure["{}_path".format(split)])),
                "blob"    : self._get_blob()
            }
            with open(os.path.join(self.configure["base_dir"], self.configure["{}_pair_path".format(split)]), "r") as f:
                pairs = f.read().split("\n")[1:-1]
                self.pairs = list(map(lambda x: x.split(','), pairs))
            with open(os.path.join(self.configure["base_dir"], self.configure["{}_annotation_path".format(split)]), "r") as f:
                annotation = f.read().split("\n")[1:-1]
            self.annotation_dict = {}
            for record in annotation:
                try:
                    k, r, c = record.split(':')
                    self.annotation_dict[k] = [eval(r), eval(c)]
                except Exception as e:
                    pass
            self.data_info["annotation_dict"] = self.annotation_dict
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
        image = Image.open(img_path, 'r').convert('RGB')
        assert image is not None, "{}, image not exists, img_path".format(img_path)
        keypoint = self.kp2tensor(self.annotation_dict[name])
        return image, keypoint

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        (src, src_kp), (trg, trg_kp) = self._get_one(pair[0]), self._get_one(pair[1])
        src, trg = self.preprocess(src), self.preprocess(trg)
        src_kp, trg_kp = src_kp.to_dense(), src_kp.to_dense()
        blob = self._get_blob(src, src_kp, trg, trg_kp)
        return blob

    def kp2tensor(self, corr) -> torch.Tensor:
        indx = [[], [], []]
        valu = []
        for i, (r, c) in enumerate(zip(corr[0], corr[1])):
            # Note: This setting is only for deepfashion.
            if r < 0 or c < 0 or r >= self.w or c + 40 >= self.h:
                continue
            indx[0].append(r)
            indx[1].append(c + 40)
            indx[2].append(i)
            valu.append(1)
        indices_list = torch.LongTensor(indx)
        tensor = torch.sparse_coo_tensor(indices_list, valu, size=(self.h, self.w, self.num_kp))
        return tensor


if __name__ == '__main__':
    pass
    # data = KeypointDataset(opt=None)
    # print(len(data[1]))
