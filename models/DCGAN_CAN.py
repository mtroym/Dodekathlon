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

from models.helpers import get_scheduler


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
        self.header = nn.Linear(self.latent_dim, init_size * init_size * self.hidden)  # BN x 4 x 4 x 1024
        self.norm = norm_layer(self.hidden, eps=1e-5, momentum=0.9)
        self.model_list = []
        for i in range(len(self.channel_list) - 1):
            in_cha = int(self.channel_list[i])
            out_cha = int(self.channel_list[i + 1])
            # print(in_cha, out_cha)
            self.model_list.append(nn.ConvTranspose2d(in_cha, out_cha, kernel_size=5, stride=2, padding=2, output_padding=1, padding_mode="zeros"))
            self.model_list.append(norm_layer(out_cha, eps=1e-5, momentum=0.9))
            if i != len(self.channel_list) - 2:
                self.model_list.append(self.activation())
            else:
                self.model_list.append(nn.Tanh())
        self.core = nn.Sequential(
            *self.model_list
        )

        self.norm.float()
        self.header.float()
        self.core.float()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        linear = self.header(inputs)
        # print(linear.shape)
        linear_reshape = linear.reshape((batch_size, self.hidden, self.size_list[0], self.size_list[0]))
        normalized = self.norm(linear_reshape)
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
        self.enter = nn.Conv2d(self.in_channel, self.channel_list[0], kernel_size=5, stride=2, padding=2, padding_mode="reflex")
        self.enter_act = self.activation(0.2)
        for i in range(len(self.channel_list) - 1):
            self.model_list.append(nn.Conv2d(self.channel_list[i], self.channel_list[i + 1], kernel_size=5, stride=2, padding=2, padding_mode="reflex"))
            self.model_list.append(self.norm(self.channel_list[i + 1]))
            self.model_list.append(self.activation(0.2))
        self.core = nn.Sequential(*self.model_list)
        self.classifier_rf = nn.Sequential(
            nn.Linear(self.channel_list[-1] * 16, num_class),
            nn.Sigmoid()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        before_core = self.enter_act(self.enter(inputs))
        score = self.core(before_core)
        # print("Score", score.shape)
        flatten = score.flatten(start_dim=1)
        # print(flatten.shape)
        discriminator_output = self.classifier_rf(flatten)
        return discriminator_output


class CANModel:
    def __init__(self, opt):
        self.name = 'PATN'
        self.opt = opt

        self.is_train = True
        self.batch_size = opt.batchSize
        self.gpu_ids = opt.gpu_ids if opt.gpu_ids else []
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and len(self.gpu_ids) > 0) else "cpu")
        self.dtype = torch.cuda.FloatTensor if self.device != torch.device("cpu") else torch.FloatTensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.discriminator = CANDiscriminator(opt.channel, num_class=2)
        self.generator = CANGenerator(latent_dim=opt.latent_dim, hidden=opt.hidden)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.schedular_D = get_scheduler(self.optimizer_D, opt)
        self.schedular_G = get_scheduler(self.optimizer_G, opt)

    def cuda(self):
        self.discriminator.type(self.dtype)
        self.generator.type(self.dtype)

    def train_batch(self, inputs: dict, loss: dict, metrics: dict) -> dict:

        real = inputs["Source"].type(self.dtype)
        real_label = inputs["Class"].type(self.dtype)
        fake_label = torch.zeros((self.opt.batchSize, 2), device=self.device).type(self.dtype)
        fake_label[:, 0] = 1

        self.optimizer_D.zero_grad()
        # train with real label for D
        pred_real = self.discriminator(real)
        print(pred_real.shape)
        err_d_real = loss["bce_loss"](pred_real, real_label)
        # train with fake label for D
        fixed_noise = torch.randn((self.opt.batchSize, self.opt.latent_dim), device=self.device)
        fake = self.generator(fixed_noise)
        pred_fake = self.discriminator(fake)
        err_d_fake = loss["bce_loss"](pred_fake, fake_label)
        err_d = err_d_fake + err_d_real
        err_d.backward(retain_graph=True)
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        pred_fake = self.discriminator(fake)
        err_g = loss["bce_loss"](pred_fake, fake_label)
        err_g.backward()
        self.optimizer_G.step()

        with torch.no_grad():
            fake = self.generator(fixed_noise).detach().cpu()

        return {"Target": fake, "Loss_G": float(err_g.mean()), "Loss_D": float(err_d.mean())}


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
