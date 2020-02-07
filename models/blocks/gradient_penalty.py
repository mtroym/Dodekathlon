# -*- coding: utf-8 -*-
"""
@Project:   pytorch-train
@File   :   gradient_penalty
@Author :   TonyMao@AILab
@Date   :   2020/2/7
@Desc   :   None
"""
import torch


def calculate_gradient_penatly(discriminator, real_imgs, fake_imgs, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    eta = torch.rand((real_imgs.size(0), 1, 1, 1)).to(device)
    eta = eta.expand((real_imgs.size(0), real_imgs.size(1), real_imgs.size(2), real_imgs.size(3)))
    interpolated = eta * (real_imgs - fake_imgs) + fake_imgs
    interpolated.requires_grad = True

    # calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

    return gradients_penalty
