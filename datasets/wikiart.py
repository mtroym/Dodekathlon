# -*- coding: utf-8 -*-
"""
@Project:   pytorch-train
@File   :   wikiart
@Author :   TonyMao@AILab
@Date   :   2020/1/17
@Desc   :   None
"""
import os

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import yaml
from PIL import Image

from datasets.utils import get_transform


class ArtsDataset:
    def __init__(self, opt, split="train"):
        self.base_dir = None
        self.image_dir = None
        self.split = split
        self.opt = opt
        # set out space params.
        self.data_info_path = os.path.join(self.opt.data_gen, self.opt.dataset + "_{}_info.pth.tar".format(split))
        self.configure = yaml.load(open(os.path.join(self.opt.configure_path, self.opt.dataset + '.yaml'), 'r'))

        for k, v in self.configure.items():
            # self.opt.__setattr__('in_channel', self.configure['channel'])
            self.opt.__setattr__(k, v)

        self.cate = self.opt.data_class["class"]
        self.subclass = self.opt.data_class["subclass"]

        # custom this from multiple datasets.
        if self.opt.dataset == "wikiart":
            self.custom_transformation = transforms.Compose([
                transforms.RandomRotation(180),
                transforms.RandomRotation(90),
            ])
            print("wikicustom transform setting..")

        self.preprocess = transforms.Compose([
            self.custom_transformation,
            get_transform(opt),
        ])

        if os.path.exists(self.data_info_path) and not self.opt.refresh:
            print("=> load dataset information from {}".format(self.data_info_path))
            self.data_info = torch.load(self.data_info_path)
        else:
            print("=> create dataset information....")
            self.data_info = {
                "blob": self._get_blob()
            }
            # ==================== training entries ====================
            split_base_dir = os.path.join(self.configure["base_dir"], self.configure["split_dir"])
            # with open(os.path.join(self.configure["base_dir"], self.configure["{}_pair_path".format(split)]), "r") as f:
            if self.opt.dataset == "wikiart":
                print("wikicustom datapath setting..")
            # annotation path.

            for cate in self.configure["data_aggregations"]:
                if cate not in self.data_info:
                    self.data_info[cate] = {}

                # ============ Make Mappings ==========
                namepath = os.path.join(split_base_dir, self.configure["data_details"][cate]["label2name"])
                with open(namepath, "r") as f:
                    data = f.read().split("\n")[:-1]
                self.data_info[cate]["name2index"] = dict()
                self.data_info[cate]["index2name"] = dict()
                for entry in data:
                    label, name = entry.split()
                    self.data_info[cate]["name2index"][name] = int(label)
                    self.data_info[cate]["index2name"][label] = name

                for data_split in ["train", "val"]:
                    path = os.path.join(split_base_dir, self.configure["data_details"][cate]["{}_list".format(data_split)])
                    with open(path, "r") as f:
                        data = f.read().split("\n")[:-1]
                        data = list(map(self.entry2data, data))
                        self.data_info[cate][data_split] = data
                self.data_info["image_dir"] = os.path.join(self.configure["base_dir"], self.configure["image_dir"])
            # store annotation.
            # ================= create checkpoint files. to load next time. ======================
            torch.save(self.data_info, self.data_info_path)
            print("=> saved dataset information from {}".format(self.data_info_path))
        for k, v in self.data_info.items():
            self.__setattr__(k, v)
        # out of save...
        if self.subclass != "all":
            subclass_label = self.get_label_index(self.cate, self.subclass)
            self.data_list = [name_label for name_label in self.data_info[self.cate][split] if name_label[1] == subclass_label]
        else:
            self.data_list = self.data_info[self.cate][split]

        print("=> Total num of {}ing pairs: {}".format(self.split, self.__len__()))

    @staticmethod
    def _get_blob(src=None, cls=None):
        return {"Source": src, "Class": cls}

    @staticmethod
    def entry2data(entry=""):
        path, class_num = entry.split(",")
        return path, int(class_num)

    def get_label_index(self, cate, name):
        return self.data_info[cate]["name2index"][name]

    def __len__(self):
        return len(self.data_list)

    def _get_one(self, idx):
        # mapping to path.
        img_path, label = self.data_list[idx]
        img_path = os.path.join(self.image_dir, img_path)
        image = Image.open(img_path, 'r').convert('RGB')
        assert image is not None, "{}, image not exists, img_path".format(img_path)
        return image, label

    def __getitem__(self, idx):
        src, cls = self._get_one(idx)
        processed_src = self.preprocess(src)
        if self.subclass != "all":
            # cls = torch.LongTensor(1)
            cls = 1
        else:
            pass
        blob = self._get_blob(processed_src, cls)
        return blob


if __name__ == '__main__':
    pass
    # data = KeypointDataset(opt=None)
    # print(len(data[1]))
