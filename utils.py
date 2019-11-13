# -*- coding: utf-8 -*-
"""
@Project:   lab.ad.glserver
@File   :   utils
@Author :   TonyMao@AILab
@Date   :   2019-09-10
@Desc   :   None
"""
import multiprocessing as mp
import os
import traceback
import uuid

import cv2
import numpy as np
import wget
import threading

from runtime_utils import tos_helper
from runtime_utils.log_helper import logger as logging

if not os.environ.get('IS_TCE_DOCKER_ENV'):
    MATERIAL_ROOT = "../media"
else:
    MATERIAL_ROOT = os.environ.get('MATERIAL_ROOT')




def master_download(queue):
    max_num = 100
    p_list = []
    while True:
        for p in p_list:
            if not p.is_alive():
                p.terminate()
                p_list.remove(p)
        if len(p_list) == max_num:
            continue
        url, path = queue.get()
        if url is None and path is None:
            while len(p_list) != 0:
                for p in p_list:
                    if not p.is_alive():
                        p.terminate()
                        p_list.remove(p)
            break
        p = mp.Process(target=download, args=(url, path,))
        # p.daemon = True
        p.start()
        p_list.append(p)


class AddressBook0:
    """
    MATERIAL_ID |> URL
                |> LOCAL_PATH
    """

    def __init__(self, task_id):
        self.task_id = task_id
        self.local_path_book = dict()
        self.dtype = dict()
        self.url_book = dict()
        self.download_queue = mp.Queue()
        self.c = mp.Process(target=master_download, args=(self.download_queue,))
        self.c.start()

    def update(self, mat_id, url, dtype="image") -> None:
        if mat_id in self.local_path_book.keys():
            return self.local_path_book[mat_id]
        # if url in self.url_book:
        #     return self.local_path_book[mat_id]
        self.dtype[mat_id] = dtype
        self.url_book[mat_id] = url
        if dtype.lower() == "image":
            local_path = os.path.join(MATERIAL_ROOT, self.task_id, str(uuid.uuid4()) + '.png')
            # download(url, local_path)
            self.download_queue.put((url, local_path))
            self.local_path_book[mat_id] = local_path
            # return local_path
        elif dtype.lower() == "video":
            local_path = os.path.join(MATERIAL_ROOT, self.task_id, str(uuid.uuid4()) + '.mp4')
            # download(url, local_path)
            self.download_queue.put((url, local_path))
            self.local_path_book[mat_id] = local_path
            # return local_path
        else:  # GS
            local_path_dict = {}
            uidd = str(uuid.uuid4())
            for comp in ["bg", "mask", "toplayer"]:
                if url[comp] is not None:
                    video = "video-url" in url[comp].keys()
                    path = os.path.join(MATERIAL_ROOT, self.task_id, uidd + "_" + comp + (".mp4" if video else ".png"))
                    url_remote = url[comp]["video-url" if video else "image-url"]
                    # download(url_remote, path)
                    self.download_queue.put((url_remote, path))
                    local_path_dict[comp] = path
            comp = "trans"
            path = os.path.join(MATERIAL_ROOT, self.task_id, uidd + "_mtx.npy")
            local_path_dict[comp] = path
            np.save(path, url[comp])
            # tran
            self.local_path_book[mat_id] = local_path_dict
            # return local_path_dict

    def update_local(self, mat_id, local_path, dtype='image'):
        self.local_path_book[mat_id] = local_path
        self.dtype[mat_id] = dtype
        return local_path

    def __getitem__(self, item):
        return self.local_path_book[item]


class AddressBook:
    """
    MATERIAL_ID |> URL
                |> LOCAL_PATH
    """

    def __init__(self, task_id):
        self.task_id = task_id
        self.local_path_book = dict()
        self.dtype = dict()
        self.url_book = dict()
        self.download_threads = []

    def update(self, mat_id, url, dtype="image") -> None:

        if mat_id in self.local_path_book.keys():
            return self.local_path_book[mat_id]
        # if url in self.url_book:
        #     return self.local_path_book[mat_id]
        self.dtype[mat_id] = dtype
        self.url_book[mat_id] = url
        if dtype.lower() == "image":
            local_path = os.path.join(MATERIAL_ROOT, self.task_id, str(uuid.uuid4()) + '.png')
            upper_f = "/".join(local_path.split("/")[:-1])
            if not os.path.exists(upper_f):
                os.makedirs(upper_f)
            # download(url, local_path)
            t = threading.Thread(target=download, args=(url, local_path))
            t.start()
            self.download_threads.append(t)
            self.local_path_book[mat_id] = local_path
            # return local_path
        elif dtype.lower() == "video":
            local_path = os.path.join(MATERIAL_ROOT, self.task_id, str(uuid.uuid4()) + '.mp4')
            upper_f: str = "/".join(local_path.split("/")[:-1])
            if not os.path.exists(upper_f):
                os.makedirs(upper_f)
            # download(url, local_path)
            t = threading.Thread(target=download, args=(url, local_path))
            t.start()
            self.download_threads.append(t)
            self.local_path_book[mat_id] = local_path
            # return local_path
        else:  # GS
            local_path_dict = {}
            uidd = str(uuid.uuid4())
            for comp in ["bg", "mask", "toplayer"]:
                if url[comp] is not None:
                    video = "video-url" in url[comp].keys()
                    path = os.path.join(MATERIAL_ROOT, self.task_id, uidd + "_" + comp + (".mp4" if video else ".png"))
                    url_remote = url[comp]["video-url" if video else "image-url"]
                    upper_f = "/".join(path.split("/")[:-1])
                    if not os.path.exists(upper_f):
                        os.makedirs(upper_f)
                    # download(url_remote, path)
                    t = threading.Thread(target=download, args=(url_remote, path))
                    t.start()
                    self.download_threads.append(t)
                    #self.download_queue.put((url_remote, path))
                    local_path_dict[comp] = path
            comp = "trans"
            path = os.path.join(MATERIAL_ROOT, self.task_id, uidd + "_mtx.npy")
            local_path_dict[comp] = path
            np.save(path, url[comp])
            # tran
            self.local_path_book[mat_id] = local_path_dict
            # return local_path_dict

    def update_local(self, mat_id, local_path, dtype='image'):
        self.local_path_book[mat_id] = local_path
        self.dtype[mat_id] = dtype
        return local_path

    def join_all(self):
        while len(self.download_threads) != 0:
            for t in self.download_threads:
                if not t.is_alive():
                    t.terminate()
                    self.download_threads.remove(t)

    def __getitem__(self, item):
        return self.local_path_book[item]


def load_effect(name):
    if name is None:
        return None
    effect_content = tos_helper.download(key='glsleffects/' + name + '.glsl').decode()
    return effect_content


def download(url, path):
    try:
        wget.download(url, out=path)
    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error("Error in url:{}".format(url))
    return path


def _gen_bg(h, w):
    length = 50
    # image = (np.random.rand(w, h, 3) * 255).astype(np.uint8)
    image = np.ones((w, h, 3), dtype=np.uint8) * 255
    for j in range(h):
        for i in range(w):
            if (int(i / length) + int(j / length)) % 2:
                image[i, j, :] = [216, 120, 200]
    return image


def _gen_materials():
    from tlglhandler.tlglobalconfigure import global_config, set_fps
    set_fps(60.0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_source/checkerboard{}fps{}x{}.mp4'.format(global_config.default_fps, global_config.canvas_w, global_config.canvas_h),
                          fourcc, global_config.default_fps, (global_config.canvas_w, global_config.canvas_h))

    # cv2.imwrite('test.png', _gen_bg(global_config.canvas_h, global_config.canvas_w))
    bg_default = _gen_bg(global_config.canvas_w, global_config.canvas_h)
    for i in range(int(10 * global_config.default_fps)):
        bg = bg_default.copy()
        cv2.putText(bg, 'Video#4',
                    (50, global_config.canvas_h // 2), cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 0, 0), thickness=8)
        cv2.putText(bg, '{}FPS#{}'.format(global_config.fps, i + 1), (50, 200), 0, fontScale=3, color=(0, 0, 0), thickness=2)
        cv2.putText(bg, '{:.2f}s'.format((i + 1) / global_config.default_fps),
                    (100, int(global_config.canvas_h / 1.5)), cv2.FONT_HERSHEY_TRIPLEX, fontScale=6, color=(0, 0, 0), thickness=10)
        # cv2.imwrite('test_source/test{}.png'.format(i), bg)
        out.write(bg)
        del bg
    out.release()


def _gen_image():
    from tlglhandler.tlglobalconfigure import global_config
    bg_default = _gen_bg(global_config.canvas_w, global_config.canvas_h)
    cv2.putText(bg_default, 'Image#3',
                (50, global_config.canvas_h // 2), cv2.FONT_HERSHEY_TRIPLEX, fontScale=4, color=(0, 0, 0), thickness=8)
    cv2.imwrite('test_source/image_placeholder3.png', bg_default)


if __name__ == '__main__':
    # print(load_effect("effect/douyin"))
    # _gen_image()
    _gen_materials()
