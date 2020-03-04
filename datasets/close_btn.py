import imghdr
import multiprocessing as mp
import os
import random
import uuid

import numpy as np
import requests
import torch
import yaml
from PIL import Image, ImageOps
from torchvision.transforms import transforms

from datasets.utils import get_transform
from tools.progbar import progbar


class CloseButton:
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

        self.btn_path = self.opt.btn_path
        if self.opt.dataset == "close_btn":
            self.transforms = transforms.Compose([
                transforms.ColorJitter()
            ])
            print("close_btn transform setting..")

        self.preprocess = transforms.Compose([
            self.transforms,
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
            with open(os.path.join(split_base_dir, "{}.txt".format(split)), "r") as f:
                self.data_list = f.read().split()
                if self.data_list[-1] == "":
                    self.data_list = self.data_list[:-1]
            self.data_info["image_dir"] = os.path.join(self.opt.base_dir, self.opt.image_dir)
            self.data_info["data_list"] = self.data_list
            # ================= create checkpoint files. to load next time. ======================
            torch.save(self.data_info, self.data_info_path)
            print("=> saved dataset information from {}".format(self.data_info_path))

        for k, v in self.data_info.items():
            self.__setattr__(k, v)

        print("=> Total num of {}ing pairs: {}".format(self.split, self.__len__()))

    @staticmethod
    def _get_blob(src=None, coor=None):
        return {"Source": src, "Coordinates": coor}

    def __len__(self):
        return len(self.data_list)

    def blend_btn(self, _img, img_size):
        w, h = img_size
        side = h
        random_scale = 1 + (random.randint(0, 100) - 50) / 200  # scale in[0.75, 1.25]
        btn_side = int(w * .12 * random_scale)
        btn_size = (btn_side, btn_side)
        btn = Image.new("L", btn_size, 128)
        coor_r = random.randint(0, h - btn_side)
        coor_c = random.randint(0, w - btn_side)
        btn_alpha = Image.blend(Image.open(open(self.btn_path, "rb")).resize(btn_size).convert(
            'RGBA').split()[-1], Image.new("L", btn_size, 0), 0.2)
        _img.paste(btn, (coor_c, coor_r), mask=btn_alpha)
        return _img, {"Coordinates": np.array([(coor_c + btn_side / 2) / side, (coor_r + btn_side / 2) / side,
                                               btn_side / side, btn_side / side])}

    def _get_one(self, idx):
        # mapping to path.
        img_path = self.data_list[idx]
        img_path = os.path.join(self.image_dir, img_path)
        image = Image.open(img_path, 'r').convert('RGB').resize((234, 416))
        image = ImageOps.expand(image, (0, 0, 416 - 234, 0), fill=(0, 0, 0))
        image, label = self.blend_btn(image, (234, 416))
        image = self.preprocess(image)
        assert image is not None, "{}, image not exists, img_path".format(img_path)
        return image, label

    def __getitem__(self, idx):
        src, cls = self._get_one(idx)
        blob = self._get_blob(src, cls)
        return blob


def gen_list():
    save_path_list = "/Users/tony/Develop/data/close_btn/"
    save_path = os.path.join(save_path_list, "img")
    gen_status = mp.Manager().dict()
    n_procs = 32
    queue = mp.Queue()
    results = mp.Queue()

    def request_download(path, image_url):
        r = requests.get(image_url)
        ext = imghdr.what(None, r.content)
        save_name = uuid.uuid1().hex + "." + (ext if ext is not None else "png")
        save_path = os.path.join(path, save_name)
        with open(save_path, "wb") as f:
            f.write(r.content)
        return save_name

    def func(queue, result):
        while not queue.empty():
            try:
                url = queue.get()
                local_data = request_download(save_path, url)
                result.put(local_data)
                pid = os.getpid()
                gen_status[pid] = 1 if not pid in gen_status.keys() else gen_status[pid] + 1
            except Exception as ex:
                # print(str(ex))
                break

    def mp_run(data_list, split, func):
        for data in data_list:
            queue.put(data)

        mp_pools = []
        for _ in range(n_procs):
            mp_t = mp.Process(target=func, args=(queue, results))
            mp_pools.append(mp_t)
            mp_t.start()

        my_bar = progbar(len(data_list), width=30)
        while True:
            sum_cnt = sum([gen_status[pid] for pid in gen_status.keys()])
            my_bar.update(sum_cnt)
            if sum_cnt == len(data_list):
                break
        with open(os.path.join(save_path_list, "{}.txt".format(split)), "w") as f:
            # f.write("\n".join(data_list))
            while not results.empty():
                f.write(results.get_nowait() + "\n")

    split = (os.sys.argv[1])
    image_url_list = "../configures/image_urls.txt"
    save_path_list = "/Users/tony/Develop/data/close_btn/"
    save_path = os.path.join(save_path_list, "img")
    with open(image_url_list, "r") as f:
        urls = f.read().split()
    train_list = urls[:int(len(urls) / 2)]
    val_list = urls[int(len(urls) / 2):]
    if split == "train":
        mp_run(train_list, split, func)
    else:
        mp_run(val_list, split, func)


def gen_labeled_data():
    btn_path = ""
    label_base_dir = "/mnt/cephfs_new_wj/lab_ad_idea/maoyiming/data/close_btn/labels"
    base_dir = "/mnt/cephfs_new_wj/lab_ad_idea/maoyiming/data/close_btn/image"
    gen_status = mp.Manager().dict()
    n_procs = 32
    queue = mp.Queue()

    def mp_run(data_list, func):
        for data in data_list:
            queue.put(data)

        mp_pools = []
        for _ in range(n_procs):
            mp_t = mp.Process(target=func, args=(queue, label_base_dir))
            mp_pools.append(mp_t)
            mp_t.start()

        my_bar = progbar(len(data_list), width=30)
        while True:
            sum_cnt = sum([gen_status[pid] for pid in gen_status.keys()])
            my_bar.update(sum_cnt)
            if sum_cnt == len(data_list):
                break

    def func(queue, label_base_path):
        while not queue.empty():
            try:
                url = queue.get()
                do_one(str(os.path.join(base_dir, url)), label_base_path)
                pid = os.getpid()
                gen_status[pid] = 1 if not pid in gen_status.keys() else gen_status[pid] + 1
            except Exception as ex:
                print(str(ex))
                break

    def blend_btn(_img, area):
        w, h = _img.size
        random_scale = 1 + (random.randint(0, 100) - 50) / 200  # scale in[0.75, 1.25]
        btn_side = int(w * .12 * random_scale)
        btn_size = (btn_side, btn_side)
        btn = Image.new("L", btn_size, random.randint(240, 255))
        coor_r = random.randint(int(area[0] * h), int(area[1] * h - btn_side))
        coor_c = random.randint(int(area[2] * w), int(area[3] * w - btn_side))

        random_idx = random.randint(0, 21)
        btn_path = os.path.join("/mnt/cephfs_new_wj/lab_ad_idea/maoyiming/code/gans/datasets/btn", "{}.png".format(random_idx))
        btn_alpha = Image.blend(Image.open(open(btn_path, "rb")).resize(btn_size).convert(
            'RGBA').split()[-1], Image.new("L", btn_size, 0), 0.2)
        _img.paste(btn, (coor_c, coor_r), mask=btn_alpha)
        return _img, [(coor_c + btn_side / 2) / w, (coor_r + btn_side / 2) / h, btn_side / w, btn_side / h]

    def do_one(image_path, label_base_path):
        print(image_path)
        image = Image.open(image_path, 'r').convert('RGB')
        image = image.resize((720, 1280))
        label_all = []
        pose_all = [[0, 0.5, 0, 0.5], [0.5, 1, 0, 0.5], [0, 0.5, 0.5, 1], [0.5, 1, 0.5, 1]]
        if random.random() < 0.333:
            image = image.point(lambda p: p * 0.80)
            for pos in pose_all:
                if random.random() < 0.5:
                    image, coor = blend_btn(image, pos)
                    label_all.append([0] + coor)
        image_blent = image
        image_blent.save(image_path.replace("image", "images"))
        label_path = os.path.join(label_base_path, image_path.split("/")[-1].split(".")[0] + ".txt")
        label_str = ""
        if len(label_all) != 0:
            label_str = "\n".join([" ".join(map(str, label)) for label in label_all])
        # print(label_str)
        with open(label_path, "w") as f:
            f.write(label_str)
            f.close()

    path_list = os.listdir(base_dir)
    mp_run(path_list, func)


if __name__ == '__main__':
    gen_labeled_data()
    # gen_list()
    # btn_path = "/Users/tony/PycharmProjects/pytorch-train/datasets/btn.png"
    # btn_temp = cv2.imread(btn_path, cv2.IMREAD_UNCHANGED)
    # btn_temp = cv2.resize(btn_temp, (50, 50))[:, :, -1]
    # # [:, :, -1]
    #
    # test_path = "/Users/tony/PycharmProjects/pytorch-train/datasets/test.jpeg"
    # test_out = "/Users/tony/PycharmProjects/pytorch-train/datasets/test_out.jpeg"
    #
    # img = Image.open(open(test_path, 'rb'), "RGB")
    # img.paste(btn_temp, (0, 0))
    # img.save(test_out)
    #
    # print(btn_temp.shape)
    # print("done")
    # handle_data(train_list,"train")
    # handle_data(val_list, "val")
