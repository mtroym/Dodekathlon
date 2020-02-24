"""
Author:         Yiming Mao - mtroym@github
Description:    Transplant from "https://github.com/robbiebarrat/art-DCGAN/blob/master/main.lua"
"""
import os
from collections import OrderedDict

import torch
import torch.nn as nn

from models.helpers import init_weights


class _Generator(nn.Module):
    def __init__(self, nz=100, ngf=160, nc=3, init_type="normal"):
        super(_Generator, self).__init__()
        self.init_type = init_type
        self.model = nn.Sequential(OrderedDict([
            # ----- layer 1
            ("conv1", nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 16, kernel_size=4, stride=1, padding=0, bias=False)),
            ("bn1", nn.BatchNorm2d(num_features=ngf * 16)),
            ("relu1", nn.ReLU(inplace=True)),

            # ----- layer 2
            ("conv2", nn.ConvTranspose2d(in_channels=ngf * 16, out_channels=ngf * 8, kernel_size=4, stride=2, padding=1, bias=False)),
            ("bn2", nn.BatchNorm2d(num_features=ngf * 8)),
            ("relu2", nn.ReLU(inplace=True)),

            # ----- layer 3
            ("conv3", nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            ("bn3", nn.BatchNorm2d(num_features=ngf * 4)),
            ("relu3", nn.ReLU(inplace=True)),

            # ----- layer 4
            ("conv4", nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            ("bn4", nn.BatchNorm2d(num_features=ngf * 2)),
            ("relu4", nn.ReLU(inplace=True)),

            # ----- layer 5
            ("conv5", nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=False)),
            ("bn5", nn.BatchNorm2d(num_features=ngf)),
            ("relu5", nn.ReLU(inplace=True)),

            ("conv6", nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False)),
            ("tanh", nn.Tanh()),
        ]))
        # self.model.apply()

    def init_param(self):
        init_weights(self.model, self.init_type)

    def forward(self, inputs):
        return self.model(inputs)


class _Discriminator(nn.Module):
    def __init__(self, ndf=40, nc=3, init_type="normal"):
        super(_Discriminator, self).__init__()
        self.init_type = init_type
        self.model = nn.Sequential(OrderedDict([
            # ------- layer 1
            ("conv1", nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1)),
            ("bn1", nn.BatchNorm2d(num_features=ndf)),
            ("lrelu1", nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            # ------- layer 2
            ("conv2", nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1)),
            ("bn2", nn.BatchNorm2d(num_features=ndf * 2)),
            ("lrelu2", nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            # ------- layer 3
            ("conv3", nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1)),
            ("bn3", nn.BatchNorm2d(num_features=ndf * 4)),
            ("lrelu3", nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            # ------- layer 4
            ("conv4", nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1)),
            ("bn4", nn.BatchNorm2d(num_features=ndf * 8)),
            ("lrelu4", nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            # ------- layer 5
            ("conv5", nn.Conv2d(in_channels=ndf * 8, out_channels=ndf * 16, kernel_size=4, stride=2, padding=1)),
            ("bn5", nn.BatchNorm2d(num_features=ndf * 16)),
            ("lrelu5", nn.LeakyReLU(negative_slope=0.2, inplace=True)),

            # ------- layer 6
            ("conv6", nn.Conv2d(in_channels=ndf * 16, out_channels=1, kernel_size=4)),
            ("sigmoid", nn.Sigmoid()),
        ]))

    def init_param(self):
        init_weights(self.model, self.init_type)

    def forward(self, inputs):
        return self.model(inputs).flatten()


class DCGANModel:
    def __init__(self, opt):
        self.name = "DCGAN model"
        self.opt = opt
        self.in_channel = opt.channel
        self.init_type = self.opt.init_type
        self.gpu_ids = opt.gpu_ids if opt.gpu_ids else []
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and len(self.gpu_ids) > 0) else "cpu")
        self.dtype = torch.cuda.FloatTensor if self.device != torch.device("cpu") else torch.FloatTensor
        self.save_dir = opt.expr_dir

        if self.opt.fine_size == 128:
            self.ndf = 40
            self.ngf = 160
            self.nz = 100

        self.discriminator = _Discriminator(ndf=self.ndf, nc=self.in_channel, init_type=self.init_type).to(self.device)
        self.generator = _Generator(nz=self.nz, ngf=self.ngf, nc=self.in_channel, init_type=self.init_type).to(self.device)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.resume_path is not None:
            pass
        # place_holders
        self.inputs = None
        self.loss = None
        self.metrics = None
        self.label: torch.Tensor() = None
        self.current_minibatch = self.opt.batchSize
        self.fixed_noise = self.gen_noise()
        self.cuda()

    def cuda(self):
        self.discriminator.cuda(self.device)
        self.generator.cuda(self.device)

    def train_discriminator(self):
        self.optimizer_d.zero_grad()
        real = self.inputs["Source"].to(self.device)
        self.label.fill_(1)
        pred_real = self.discriminator(real)
        loss_d_real = self.loss["bce_loss_d"](pred_real, self.label)
        loss_d_real.backward()

        noise = self.gen_noise()
        fake = self.generator(noise)
        pred_fake = self.discriminator(fake)
        self.label.fill_(0)
        loss_d_fake = self.loss["bce_loss_d"](pred_fake, self.label)
        loss_d_fake.backward()

        self.optimizer_d.step()
        return float(loss_d_fake.detach().mean() + loss_d_real.detach().mean())

    def train_generator(self):
        self.optimizer_g.zero_grad()
        self.label.fill_(1)
        noise = self.gen_noise()
        fake = self.generator(noise)
        pred_fake = self.discriminator(fake)
        loss_g = self.loss["bce_loss_g"](pred_fake, self.label)
        loss_g.backward()
        self.optimizer_g.step()
        return float(loss_g.detach().mean())

    def gen_noise(self):
        if self.opt.fine_size == 128:
            noise = torch.randn((self.current_minibatch, self.nz, 1, 1)).to(self.device)
        else:
            raise NotImplementedError("The fine size is not supported")
        return noise

    def train_batch(self, inputs: dict, loss: dict, metrics: dict, niter: int = 0, epoch: int = 0) -> dict:
        self.inputs = inputs
        self.loss = loss if self.loss is None else self.loss
        self.metrics = metrics if self.metrics is None else self.metrics
        self.current_minibatch = inputs["Source"].shape[0]
        self.label = torch.full((self.current_minibatch,), 1).to(self.device)
        loss_d = self.train_discriminator()
        loss_g = self.train_generator()
        with torch.no_grad():
            fake = self.generator(self.fixed_noise)
        store_val = {"vis": {"Target": fake,
                             "Source": inputs["Source"]},
                     "loss": {"loss_d": loss_d,
                              "loss_g": loss_g}}
        self.save_model(epoch, store=store_val)
        return store_val

    def save_model(self, epoch, store):
        store_dict = {
            "epoch": epoch,
            "model_state_dict": {
                "discriminator_model_state_dict": self.discriminator.state_dict(),
                "generator_model_state_dict": self.generator.state_dict()
            },
            "optimizer_state_dict": {
                "discriminator_optimizer_state_dict": self.optimizer_d.state_dict(),
                "generator_optimizer_state_dict": self.optimizer_d.state_dict()
            }
        }
        store_dict.update(store)
        torch.save(store_dict, os.path.join(self.opt.expr_dir, "epoch_{}.pth".format(epoch)))
        torch.save(store_dict, os.path.join(self.opt.expr_dir, "latest.pth".format(epoch)))

    def load_model(self, store_path):
        store_dict = torch.load(store_path)
        self.discriminator.load_state_dict(store_dict["model_state_dict"]["discriminator_model_state_dict"])
        self.generator.load_state_dict(store_dict["model_state_dict"]["generator_model_state_dict"])
        self.optimizer_d.load_state_dict(store_dict["optimizer_state_dict"]["discriminator_optimizer_state_dict"])
        self.optimizer_g.load_state_dict(store_dict["optimizer_state_dict"]["generator_optimizer_state_dict"])


if __name__ == '__main__':
    bs = 2
    w, h = 128, 128
    num_z = 100
    image = torch.rand((bs, 3, w, h))
    z = torch.rand((bs, num_z, 1, 1))
    g = _Generator()
    d = _Discriminator()
    fak = g(z)
    print(fak.shape)
    print(d(fak).shape)
