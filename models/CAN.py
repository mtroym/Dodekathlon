# -*- coding: utf-8 -*-
"""
@Project:   pytorch-train
@File   :   DCGAN_CAN
@Author :   TonyMao@AILab
@Date   :   2020/1/16
@Desc   :   None
"""

import torch

from models.CANx256 import CANGenerator, CANDiscriminator
from models.CANx64 import Generator, Discriminator, nz
from models.blocks.gradient_penalty import calculate_gradient_penatly
from models.helpers import init_weights


class CANModel:
    def __init__(self, opt):
        self.name = 'Creative Adversarial Network'
        self.opt = opt

        self.is_train = True
        self.batch_size = opt.batchSize
        self.gpu_ids = opt.gpu_ids if opt.gpu_ids else []
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and len(self.gpu_ids) > 0) else "cpu")
        self.dtype = torch.cuda.FloatTensor if self.device != torch.device("cpu") else torch.FloatTensor
        self.save_dir = opt.expr_dir
        if self.opt.fine_size == 256:
            self.discriminator = CANDiscriminator(opt.channel, num_class=2).to(self.device)
            self.generator = CANGenerator(latent_dim=opt.latent_dim, hidden=opt.hidden).to(self.device)
            self.fixed_noise = torch.randn(self.batch_size, self.opt.latent_dim, 1, 1).to(self.device)
        elif self.opt.fine_size == 64:
            self.discriminator = Discriminator(0)
            self.generator = Generator(0, u="up+conv")
            self.fixed_noise = torch.randn((self.batch_size, nz, 2, 2)).to(self.device)
        else:
            self.discriminator = Discriminator(0)
            self.generator = Generator(0, u="up+conv")
            self.fixed_noise = torch.randn((self.batch_size, nz, 2, 2)).to(self.device)
            # raise Exception("not support size")
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        # self.schedular_D = get_scheduler(self.optimizer_D, opt)
        # self.schedular_G = get_scheduler(self.optimizer_G, opt)
        self.time_g = 10
        self.time_d = 10

        init_weights(self.discriminator)
        init_weights(self.generator)
        self.cuda()

    def cuda(self):
        self.discriminator.type(self.dtype)
        self.generator.type(self.dtype)

    def gen_random_noise(self, current_minibatch):
        random_noise = None
        if self.opt.fine_size == 256:
            random_noise = torch.randn((current_minibatch, self.opt.latent_dim, 1, 1)).to(self.device)
        elif self.opt.fine_size == 64:
            # random_noise = torch.randn((current_minibatch, nz, 1, 1)).to(self.device)
            random_noise = torch.softmax(torch.randn((current_minibatch, nz * 4)), dim=1).to(self.device)
            random_noise = random_noise.view((current_minibatch, nz, 2, 2))
        return random_noise

    def train_batch(self, inputs: dict, loss: dict, metrics: dict, niter: int = 0) -> dict:
        real = inputs["Source"].type(self.dtype)
        # real_label = inputs["Class"].type(self.dtype)
        # fake_label = torch.zeros((self.batch_size,), device=self.device).type(self.dtype)

        current_minibatch = real.shape[0]
        label = torch.full((current_minibatch,), 1).to(self.device)
        err_d_sum = 0
        self.optimizer_D.zero_grad()

        err_d = 0
        for _ in range(self.time_d):
            # train with real label for D
            random_noise = self.gen_random_noise(current_minibatch)
            fake = self.generator(random_noise)
            pred_real = self.discriminator(real)
            pred_fake = self.discriminator(fake)
            label.fill_(1)
            gradient_penalty = calculate_gradient_penatly(self.discriminator, real.data, fake.data, self.device)
            err_d_real = loss["bce_loss"](pred_real, label) + gradient_penalty
            err_d_real.backward(retain_graph=True)
            label.fill_(0)
            err_d_fake = loss["bce_loss"](pred_fake, label)
            err_d_fake.backward()
        self.optimizer_D.step()
        err_d_mean = err_d / self.time_d
        # for _ in range(self.time_g):

        self.optimizer_G.zero_grad()
        random_noise = self.gen_random_noise(current_minibatch)
        fake = self.generator(random_noise)
        pred_fake = self.discriminator(fake)
        label.fill_(0)
        err_g = loss["bce_loss"](pred_fake, label)
        # err_g = - torch.mean(pred_fake)
        err_g.backward()
        self.optimizer_G.step()

        with torch.no_grad():
            fake = self.generator(self.fixed_noise)

        return {
            "vis": {"Target": fake,
                    "Source": inputs["Source"]},
            "loss": {"Loss_G": err_g,
                     "Loss_D": err_d_mean,
                     # "D_fake": float(err_d_fake.mean().item()),
                     # "D_real": float(err_d_real.mean().item()),
                     # "Time_G": self.time_g
                     }
        }


if __name__ == '__main__':
    model = Generator(0, "up+conv")
    random_noise = torch.randn((10, nz, 1, 1))
    res = model(random_noise)
    print(res.shape)

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
