# -*- coding: utf-8 -*-
"""
@Project:   pytorch-train
@File   :   DCGAN_CAN
@Author :   TonyMao@AILab
@Date   :   2020/1/16
@Desc   :   None
"""
import os

import torch
from torch import nn

from models.helpers import get_scheduler, init_weights


class CANGenerator(nn.Module):
    def __init__(self, latent_dim, hidden, norm_layer=nn.BatchNorm2d, config=None):
        super(CANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.activation = nn.ReLU
        if config is None:
            self.size_list = [4, 8, 16, 32, 64, 128, 256]
            self.channel_list = [hidden, hidden / 2, hidden / 4, hidden / 8, hidden / 8, hidden / 16, 3]
        else:
            self.size_list = config["size_list"]
            self.channel_list = config["channel_list"]

        # input to the network
        init_size = self.size_list[0]
        # self.header = nn.Linear(self.latent_dim, init_size * init_size * self.hidden, bias=False)  # BN x 4 x 4 x 1024
        self.header = nn.ConvTranspose2d(self.latent_dim, out_channels=self.hidden,
                                         kernel_size=init_size, bias=False)  # BN x 4 x 4 x 1024
        self.norm = norm_layer(self.hidden, eps=1e-5, momentum=0.9)
        self.model_list = []
        for i in range(len(self.channel_list) - 1):
            in_cha = int(self.channel_list[i])
            out_cha = int(self.channel_list[i + 1])
            # print(in_cha, out_cha)
            self.model_list.append(nn.ConvTranspose2d(in_cha, out_cha, kernel_size=4, stride=2, padding=1, padding_mode="zeros", bias=False))
            self.model_list.append(norm_layer(out_cha, eps=1e-5, momentum=0.9))
            if i != len(self.channel_list) - 2:
                self.model_list.append(self.activation(inplace=True))
            else:
                self.model_list.append(nn.Tanh())
        self.core = nn.Sequential(
            *self.model_list
        )

        self.norm.float()
        self.header.float()
        self.core.float()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # batch_size = inputs.shape[0]
        head = self.header(inputs)
        # print(linear.shape)
        # linear_reshape = linear.reshape((batch_size, self.hidden, self.size_list[0], self.size_list[0]))
        normalized = self.norm(head)
        # print(normalized.shape)
        activated = self.activation(inplace=True)(normalized)
        fake_image = self.core(activated)
        return fake_image


class CANDiscriminator(nn.Module):
    def __init__(self, in_channel=3, num_class=2, norm_layer=nn.BatchNorm2d, config=None):
        super(CANDiscriminator, self).__init__()
        self.activation = nn.LeakyReLU
        self.channel_list = [32, 64, 128, 256, 512, 512]
        self.norm = norm_layer
        self.in_channel = in_channel
        self.model_list = []
        self.enter = nn.Conv2d(self.in_channel, self.channel_list[0], kernel_size=4, stride=2, padding=1, padding_mode="reflex", bias=False)
        self.enter_act = self.activation(0.2)
        for i in range(len(self.channel_list) - 1):
            self.model_list.append(nn.Conv2d(self.channel_list[i], self.channel_list[i + 1],
                                             kernel_size=4, stride=2, padding=1, padding_mode="reflex", bias=False))
            self.model_list.append(self.norm(self.channel_list[i + 1]))
            self.model_list.append(self.activation(0.2, inplace=True))
        self.core = nn.Sequential(*self.model_list)
        self.classifier_rf = nn.Sequential(
            nn.Conv2d(self.channel_list[-1], 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        before_core = self.enter_act(self.enter(inputs))
        score = self.core(before_core)
        # print("Score", score.shape)
        discriminator_output = self.classifier_rf(score)
        return discriminator_output.flatten()


nz = 100  # Size of z latent vector (i.e. size of generator input)
ngf = 64  # Size of feature maps in generator
nc = 3
ndf = 64


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        random_noise = torch.rand_like(input_data, requires_grad=False) * 0.001
        return self.main(input_data + random_noise).flatten()


class CANModel:
    def __init__(self, opt):
        self.name = 'Creative Adversarial Network'
        self.opt = opt

        self.is_train = True
        self.batch_size = opt.batchSize
        self.gpu_ids = opt.gpu_ids if opt.gpu_ids else []
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and len(self.gpu_ids) > 0) else "cpu")
        self.dtype = torch.cuda.FloatTensor if self.device != torch.device("cpu") else torch.FloatTensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if self.opt.fine_size == 256:
            self.discriminator = CANDiscriminator(opt.channel, num_class=2).to(self.device)
            self.generator = CANGenerator(latent_dim=opt.latent_dim, hidden=opt.hidden).to(self.device)
            self.fixed_noise = torch.randn(self.batch_size, self.opt.latent_dim, 1, 1).to(self.device)
        elif self.opt.fine_size == 64:
            self.discriminator = Discriminator(0)
            self.generator = Generator(0)
            self.fixed_noise = torch.randn((self.batch_size, nz, 1, 1)).to(self.device)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        # self.schedular_D = get_scheduler(self.optimizer_D, opt)
        # self.schedular_G = get_scheduler(self.optimizer_G, opt)

        init_weights(self.discriminator)
        init_weights(self.generator)
        self.cuda()

    def cuda(self):
        self.discriminator.type(self.dtype)
        self.generator.type(self.dtype)

    def train_batch(self, inputs: dict, loss: dict, metrics: dict) -> dict:
        real = inputs["Source"].type(self.dtype)
        # real_label = inputs["Class"].type(self.dtype)
        # fake_label = torch.zeros((self.batch_size,), device=self.device).type(self.dtype)

        current_minibatch = real.shape[0]

        self.optimizer_G.zero_grad()
        random_noise = None
        label = torch.full((current_minibatch,), 1).to(self.device)
        if self.opt.fine_size == 256:
            random_noise = torch.randn((current_minibatch, self.opt.latent_dim, 1, 1)).to(self.device)
        elif self.opt.fine_size == 64:
            random_noise = torch.randn((current_minibatch, nz, 1, 1)).to(self.device)
        fake = self.generator(random_noise)
        pred_fake = self.discriminator(fake)
        label.fill_(1)
        err_g = loss["bce_loss"](pred_fake, label)
        err_g.backward()
        err_g_d = float(err_g.mean().item())
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        # train with real label for D
        pred_real = self.discriminator(real)
        label.fill_(1)
        err_d_real = loss["bce_loss"](pred_real, label)
        err_d_real.backward()
        err_d = float(err_d_real.mean().item())

        # train with fake label for D
        pred_fake = self.discriminator(fake.detach())
        label.fill_(0)
        err_d_fake = loss["bce_loss"](pred_fake, label)
        err_d_fake.backward()
        err_d += float(err_d_fake.mean().item())
        # err_d.backward()
        self.optimizer_D.step()

        with torch.no_grad():
            fake = self.generator(self.fixed_noise).detach().cpu()
            print(fake.shape)

        return {"Target": fake, "Loss_G": err_g_d, "Loss_D": err_d}


if __name__ == '__main__':
    model = CANModel(None)
"""
    import matplotlib.pyplot as plt
    import numpy as np
    torch.random.manual_seed(np.random.random(1))
    generator = CANGenerator(latent_dim=100, hidden=1024, norm_layer=nn.BatchNorm2d)
    discriminator = CANDiscriminator(in_channel=3, num_class=2, norm_layer=nn.BatchNorm2d)
    init_weights(generator, "normal")
    init_weights(discriminator, "normal")
    z = torch.rand((2, 100))
    print(z.shape)
    y = generator(z)
    y_data = (y.detach().numpy()[0].transpose([1, 2, 0]) + 1) * 128
    plt.imshow(np.array(y_data, dtype='uint8'))
    plt.show()
    y_data = (y.detach().numpy()[1].transpose([1, 2, 0]) + 1) * 128
    plt.imshow(np.array(y_data, dtype='uint8'))
    plt.show()
    cv2.imwrite("test1.png", np.array(y_data, dtype='uint8'))
    fake = (torch.rand((3, 256, 256)))
    y_data = (fake.detach().numpy().transpose([1, 2, 0])) * 255
    plt.imshow(np.array(y_data, dtype='uint8'))
    cv2.imwrite("test1.png", np.array(y_data, dtype='uint8'))
    plt.show()
    res = discriminator(y)
    predict = torch.softmax(res, dim=-1)
    print(predict)
"""
