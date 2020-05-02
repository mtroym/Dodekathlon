"""
Author:         Yiming Mao - mtroym@github
Description:    Transplant from "https://github.com/xunhuang1995/AdaIN-style/blob/master/train.lua"
"""

import functools
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import vgg19

from datasets.utils import denorm
from models.blocks import AdaptiveInstanceNorm2d
from models.blocks.vgg import rename_sequential
from models.helpers import init_weights


class _Encoder(nn.Module):
    def __init__(self, pretrained=True, init_type="normal", endlayer='relu4_1', feature_hook=None):
        super(_Encoder, self).__init__()
        self.init_type = init_type
        self.pretrained = pretrained
        self.feature_hook = ["relu1_1", "relu2_1", "relu3_1", "relu4_1"] if feature_hook is None else feature_hook
        self.core = nn.Sequential()
        backbone = vgg19(pretrained=pretrained, progress=True)
        feature_extractor = rename_sequential(backbone.features)
        for name, layer in feature_extractor.named_children():
            self.core.add_module(name, layer)
            if name == endlayer:
                break
        idx = -1
        while not hasattr(self.core[idx], "out_channels"):
            idx -= 1
        self.out_channels = self.core[idx].out_channels

    def init_param(self) -> None:
        if self.pretrained:
            return
        init_weights(self.model, self.init_type)

    def frozen(self):
        for param in self.core.parameters():
            param.requires_grad = False

    def forward(self, inputs, feature_hook=None):
        if feature_hook is None:
            feature_hook = self.feature_hook
        results = OrderedDict()
        for name, layer in self.core.named_children():
            inputs = layer(inputs)
            if name in feature_hook:
                results[name] = inputs
        return results


class _Decoder(nn.Module):
    def __init__(self, enc: nn.Module, activation="relu", remove_idx=-1):
        super(_Decoder, self).__init__()
        core_list = []
        nonlinear_act = nn.ReLU
        if activation == "lrelu":
            nonlinear_act = functools.partial(nn.LeakyReLU, negative_slope=0.2)
        for name, layer in enc.core.named_children():
            if 'conv' in name:
                in_channels, out_channels = layer.in_channels, layer.out_channels
                core_list.append((activation + name.replace("conv", ""),
                                  nonlinear_act(inplace=True)))
                # core_list.append(("in{}".format(name.replace("conv", "")),
                #                   nn.InstanceNorm2d(num_features=in_channels)))
                core_list.append(("conv{}".format(name.replace("conv", "")),
                                  nn.Conv2d(out_channels, in_channels, kernel_size=3, stride=1)))
                core_list.append(("pad{}".format(name.replace("conv", "")),
                                  nn.ReflectionPad2d(padding=(1, 1, 1, 1))))

            if 'pool' in name:
                core_list.append(("up{}".format(name.replace("pool", "")),
                                  nn.UpsamplingNearest2d(scale_factor=2)))

        self.core = rename_sequential(nn.Sequential(OrderedDict(reversed(core_list))))
        # print(self)

    def forward(self, inputs) -> torch.Tensor:
        return self.core(inputs)


class AdaIN:
    def __init__(self, opt):
        self.name = "AdaIN-Style model"
        self.opt = opt
        self.in_channel = opt.channel
        self.init_type = self.opt.init_type
        self.gpu_ids = opt.gpu_ids if opt.gpu_ids else []
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and len(self.gpu_ids) > 0) else "cpu")
        self.dtype = torch.cuda.FloatTensor if self.device != torch.device("cpu") else torch.FloatTensor
        self.save_dir = opt.expr_dir
        self.encoder = _Encoder()
        if self.opt.freeze_enc:
            self.encoder.frozen()
        self.decoder = _Decoder(self.encoder)
        self.adain = AdaptiveInstanceNorm2d(num_features=self.encoder.out_channels)
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.resume_path is not None:
            pass
        # place_holders
        self.inputs = None
        self.loss = None
        self.metrics = None
        self.current_minibatch = self.opt.batchSize
        if self.opt.resume_path is not None:
            self.load_model(self.opt.resume_path)
        self.cuda()

    def cuda(self):
        if torch.cuda.is_available():
            self.encoder.cuda(self.device)
            self.decoder.cuda(self.device)
            self.adain.cuda(self.device)

    def train_decoder(self, content, style, alpha=1.0):
        self.optimizer.zero_grad()

        # find all the features with frozen VGG
        style_features = self.encoder(style)
        content_latent = self.encoder(content, feature_hook=["relu4_1"])

        # Find adain
        trans_content = self.adain(content=content_latent["relu4_1"],
                                   style=style_features["relu4_1"])
        interpolate_latent = (1.0 - alpha) * content_latent["relu4_1"] + \
                             alpha * trans_content

        transferred_image = self.decoder(interpolate_latent)
        transferred_features = self.encoder(transferred_image)

        c_loss = self.loss["content_loss"](transferred_features["relu4_1"], interpolate_latent)
        s_loss = self.loss["style_loss"](transferred_features, style_features)
        smooth_reg = self.loss["smooth_reg"](transferred_image)
        loss = c_loss.mean() + s_loss.mean() + smooth_reg.mean()
        loss.backward()
        self.optimizer.step()
        return c_loss, s_loss, smooth_reg, transferred_image

    def train_batch(self, inputs: dict, loss: dict, metrics: dict, niter: int = 0, epoch: int = 0) -> dict:
        self.inputs = inputs
        self.loss = loss if self.loss is None else self.loss
        self.metrics = metrics if self.metrics is None else self.metrics
        self.current_minibatch = inputs["Source"].shape[0]
        c_loss, s_loss, smooth_reg, transferred_image = self.train_decoder(inputs["Source"].to(self.device),
                                                                           inputs["Style"].to(self.device))
        store_val = {"vis": {"Target": denorm(transferred_image, self.device, to_board=True),
                             "Source": denorm(inputs["Source"], self.device, to_board=True),
                             "Style": denorm(inputs["Style"], self.device, to_board=True)},
                     "loss": {"loss_content": c_loss,
                              "loss_style": s_loss,
                              "smooth_reg": smooth_reg,
                              }
                     }
        if epoch % 30 == 5:
            self.save_model(epoch, store=store_val)
        return store_val

    def predict_batch(self, inputs: dict, loss=None, metrics=None, niter=None, epoch=None):
        self.current_minibatch = self.opt.batchSize
        return {
            "vis": {"Target": None},
            "loss": {}
        }

    def save_model(self, epoch, store):
        store_dict = {
            "epoch": epoch,
            "model_state_dict": {
                "decoder_model_state_dict": self.decoder.state_dict(),
            },
            "optimizer_state_dict": {
                "decoder_optimizer_state_dict": self.optimizer.state_dict(),
            }
        }
        store_dict.update(store)
        torch.save(store_dict, os.path.join(self.opt.expr_dir, "epoch_{}.pth".format(epoch)))
        torch.save(store_dict, os.path.join(self.opt.expr_dir, "latest.pth".format(epoch)))

    def load_model(self, store_path, no_opt=False):
        store_dict = torch.load(store_path)
        self.decoder.load_state_dict(store_dict["model_state_dict"]["decoder_model_state_dict"])
        if no_opt:
            return
        self.optimizer.load_state_dict(store_dict["optimizer_state_dict"]["decoder_optimizer_state_dict"])


if __name__ == '__main__':
    bs = 10
    w, h = 128, 128
    image = torch.rand((bs, 3, w, h))
    # g = _Generator_ResizeConv()
    e = _Encoder()
    d = _Decoder(e)
    adain = AdaptiveInstanceNorm2d(e.out_channels)
    te = adain(e(image)["relu4_1"], e(image)["relu4_1"])
    print(d)
    print(d(te).shape)
    # print(e(image).shape)
    # print(d(e(image)).shape)
    # print(.out_channels)
    # fak = g(z)
    # print(fak.shape)
    # print(d(fak).shape)
