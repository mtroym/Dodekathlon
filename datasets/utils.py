# -*- coding: utf-8 -*-
"""
@Project:   pytorch-train
@File   :   utils
@Author :   TonyMao@AILab
@Date   :   2019/11/12
@Desc   :   None
"""
import torchvision.transforms as transforms
from PIL import Image


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fine_size))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fine_size))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fine_size)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.load_size)))
        transform_list.append(transforms.RandomCrop(opt.fine_size))
        # pass
    elif opt.resize_or_crop == 'scale_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale(img, opt.load_size)
        ))
        transform_list.append(transforms.RandomCrop(opt.fine_size))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)


def __scale_height(img, target_height):
    ow, oh = img.size
    if oh == target_height:
        return img
    w = int(target_height * ow / oh)
    h = target_height
    return img.resize((w, h), Image.BICUBIC)

def __scale(img, target_side):
    ow, oh = img.size
    if ow < oh:
        return __scale_width(img, target_side)
    else:
        return __scale_height(img, target_side)
