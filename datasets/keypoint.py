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
        # set out space params.

        self.data_info_path = os.path.join(self.opt.data_gen, self.opt.dataset + "_{}_info.pth.tar".format(split))
        self.configure = yaml.load(open(os.path.join(self.opt.configure_path, self.opt.dataset + '.yaml'), 'r'))

        self.num_kp = self.configure['keypoint']
        self.opt.__setattr__('in_channel', self.configure['channel'])
        self.opt.__setattr__('keypoint', self.configure['keypoint'])

        # custom this from multiple datasets.
        if self.opt.dataset == "deepfashion256":
            self.h, self.w = 256, 256
            self.transforms = transforms.Pad((40, 0, 40, 0), padding_mode='edge')
        elif self.opt.dataset == "deepfashion512":
            self.h, self.w = 512, 512
            self.transforms = transforms.Pad((80, 0, 80, 0), padding_mode='edge')

        self.preprocess = transforms.Compose([
            # self.transforms,  # custom pre-processing.
            get_transform(opt),
        ])

        # data_path = os.path.join(self.configure["base_dir"], self.configure["{}_pair_path".format(split)])
        if os.path.exists(self.data_info_path) and not self.opt.refresh:
            print("=> load dataset information from {}".format(self.data_info_path))
            self.data_info = torch.load(self.data_info_path)
        else:
            print("=> create dataset information....")
            self.data_info = {
                "blob": self._get_blob()
            }
            # training entries
            with open(os.path.join(self.configure["base_dir"], self.configure["{}_pair_path".format(split)]), "r") as f:
                pairs = f.read().split("\n")[1:-1]
                self.pairs = list(map(lambda x: x.split(','), pairs))

            # annotation path.
            with open(os.path.join(self.configure["base_dir"], self.configure["{}_annotation_path".format(split)]), "r") as f:
                annotation = f.read().split("\n")[1:-1]

            # store annotation.
            self.annotation_dict = {}
            for record in annotation:
                k, r, c = record.split(':')
                self.annotation_dict[k] = [eval(r), eval(c)]

            # build path dict.
            data_path = self.configure["data_dir"]
            self.path_dict = {}
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    real_path = os.path.join(root, file)
                    full_path = os.path.join(root, file[:4] + file[5:])
                    key = "fashion" + full_path.replace(data_path, '').replace('/', '').replace("id_", "id")
                    self.path_dict[key] = real_path

            # create checkpoint files. to load next time.
            self.data_info["path_dict"] = self.path_dict
            self.data_info["annotation_dict"] = self.annotation_dict
            self.data_info["pairs"] = self.pairs
            self.data_info["total_num"] = len(self.data_info["pairs"])
            torch.save(self.data_info, self.data_info_path)
            print("=> saved dataset information from {}".format(self.data_info_path))
        for k, v in self.data_info.items():
            self.__setattr__(k, v)

        print("=> Total num of {}ing pairs: {}".format(self.split, self.__len__()))

    def _mapping_image2kp(self, image_path):
        return os.path.join(self.configure["base_dir"],
                            self.configure["{}_keypoint_path".format(self.split)],
                            image_path.split("/")[-1] + ".npy")

    @staticmethod
    def _get_blob(src=None, kp0=None, trg=None, kp1=None):
        return {"Source": src, "SourceKP": kp0, "Target": trg, "TargetKP": kp1}

    def __len__(self):
        return len(self.pairs)

    def _get_one(self, name):
        # mapping to path.
        img_path = self.path_dict[name]
        image = Image.open(img_path, 'r').convert('RGB')
        assert image is not None, "{}, image not exists, img_path".format(img_path)
        keypoint = self.kp2tensor(self.annotation_dict[name])
        return image, keypoint

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        (src, src_kp), (trg, trg_kp) = self._get_one(pair[0]), self._get_one(pair[1])
        src, trg = self.preprocess(src), self.preprocess(trg)
        src_kp, trg_kp = src_kp.to_dense().float(), src_kp.to_dense().float()
        src_kp, trg_kp = src_kp.permute([-1, 0, 1]), trg_kp.permute([-1, 0, 1])
        blob = self._get_blob(src, src_kp, trg, trg_kp)
        return blob

    def kp2tensor(self, corr) -> torch.Tensor:
        indx = [[], [], []]
        valu = []
        for i, (r, c) in enumerate(zip(corr[0], corr[1])):
            # Note: This setting is only for deepfashion256.
            if self.opt.dataset == "deepfashion256":
                if r < 0 or c < 0 or r >= self.w or c + 40 >= self.h:
                    continue
                indx[0].append(r)
                indx[1].append(c + 40)
                indx[2].append(i)
                valu.append(1)
            else:
                raise NotImplementedError("The dataset has not implemented the kp2tensor method.")
        indices_list = torch.LongTensor(indx)
        tensor = torch.sparse_coo_tensor(indices_list, valu, size=(self.h, self.w, self.num_kp))
        return tensor


if __name__ == '__main__':
    pass
    # data = KeypointDataset(opt=None)
    # print(len(data[1]))
