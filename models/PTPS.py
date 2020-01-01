import torch.nn as nn
import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter

from modules.dense_motion_module import DenseMotionModule, IdentityDeformation
from modules.movement_embedding import MovementEmbeddingModule

from modules.util import Encoder as MonkeyEncoder
from modules.util import Decoder as MonkeyDecoder
from modules.util import ResBlock3D as MonkeyResBlock3D

from .helpers import get_scheduler

import yaml
import os
import numpy as np
import cv2

class DownBlock3D(nn.Module):
    """
    Simple block for processing video (encoder).
    """

    def __init__(self, in_features, out_features, norm=False, kernel_size=4):
        super(DownBlock3D, self).__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features,
                              kernel_size=(1, kernel_size, kernel_size))
        if norm:
            self.norm = nn.InstanceNorm3d(out_features, affine=True)
        else:
            self.norm = None

    def forward(self, x):
        out = x
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        out = F.leaky_relu(out, 0.2)
        out = F.avg_pool3d(out, (1, 2, 2))
        return out


class PTPSDiscriminator(nn.Module):
    """
    PTPSDiscriminator similar to Pix2Pix
    """

    def __init__(self, num_channels=3, num_kp=10, kp_variance=0.01, scale_factor=1,
                 block_expansion=64, num_blocks=4, max_features=512, kp_embedding_params=None):
        super(PTPSDiscriminator, self).__init__()

        if kp_embedding_params is not None:
            self.kp_embedding = MovementEmbeddingModule(num_kp=num_kp, kp_variance=kp_variance,
                                                        num_channels=num_channels,
                                                        **kp_embedding_params)
            embedding_channels = self.kp_embedding.out_channels
        else:
            self.kp_embedding = None
            embedding_channels = 0

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3D(
                num_channels + embedding_channels if i == 0 else min(max_features, block_expansion * (2 ** i)),
                min(max_features, block_expansion * (2 ** (i + 1))),
                norm=(i != 0),
                kernel_size=4))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv3d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x, kp_driving, kp_source):
        out_maps = [x]
        if self.scale_factor != 1:
            x = F.interpolate(x, scale_factor=(1, self.scale_factor, self.scale_factor))

        if self.kp_embedding:
            heatmap = self.kp_embedding(x, kp_driving, kp_source)
            out = torch.cat([x, heatmap], dim=1)
        else:
            out = x
        for down_block in self.down_blocks:
            out_maps.append(down_block(out))
            out = out_maps[-1]
        out = self.conv(out)
        out_maps.append(out)
        return out_maps

class PTPSGenerator(nn.Module):
    def __init__(self, num_channels, num_kp, kp_variance, block_expansion,
                 max_features, num_blocks, num_refinement_blocks,
                 dense_motion_params=None, kp_embedding_params=None,
                 interpolation_mode='nearest'):
        super(PTPSGenerator, self).__init__()

        self.appearance_encoder = MonkeyEncoder(block_expansion,
                                                in_features=num_channels,
                                                max_features=max_features,
                                                num_blocks=num_blocks)
        if kp_embedding_params is not None:
            self.kp_embedding_module = MovementEmbeddingModule(num_kp=num_kp,
                                                               kp_variance=kp_variance,
                                                               num_channels=num_channels,
                                                               **kp_embedding_params)
            embedding_features = self.kp_embedding_module.out_channels
        else:
            self.kp_embedding_module = None
            embedding_features = 0

        if dense_motion_params is not None:
            self.dense_motion_module = DenseMotionModule(num_kp=num_kp, kp_variance=kp_variance,
                                                         num_channels=num_channels,
                                                         **dense_motion_params)
        else:
            self.dense_motion_module = IdentityDeformation()

        self.video_decoder = MonkeyDecoder(block_expansion=block_expansion, in_features=num_channels,
                                           out_features=num_channels, max_features=max_features, num_blocks=num_blocks,
                                           additional_features_for_block=embedding_features,
                                           use_last_conv=False)

        self.refinement_module = torch.nn.Sequential()
        in_features = block_expansion + num_channels + embedding_features

        for i in range(num_refinement_blocks):
            self.refinement_module.add_module('r' + str(i),
                                              MonkeyResBlock3D(in_features,
                                                               kernel_size=(1, 3, 3),
                                                               padding=(0, 1, 1)))
        self.refinement_module.add_module('conv-last', nn.Conv3d(in_features, num_channels, kernel_size=1, padding=0))
        self.interpolation_mode = interpolation_mode

    def deform_input(self, inp, deformations_absolute):
        bs, d, h_old, w_old, _ = deformations_absolute.shape
        _, _, _, h, w = inp.shape
        deformations_absolute = deformations_absolute.permute(0, 4, 1, 2, 3)
        deformation = F.interpolate(deformations_absolute, size=(d, h, w), mode=self.interpolation_mode)
        deformation = deformation.permute(0, 2, 3, 4, 1)
        deformed_inp = F.grid_sample(inp, deformation)
        return deformed_inp

    def forward(self, source_image, kp_driving, kp_source):
        appearance_skips = self.appearance_encoder(source_image)

        deformations_absolute = self.dense_motion_module(source_image=source_image,
                                                         kp_driving=kp_driving,
                                                         kp_source=kp_source)
        deformed_skips = [self.deform_input(skip, deformations_absolute) for skip in appearance_skips]

        if self.kp_embedding_module is not None:
            d = kp_driving['mean'].shape[1]
            movement_embeding = self.kp_embedding_module(source_image=source_image,
                                                         kp_driving=kp_driving,
                                                         kp_source=kp_source)
            kp_skips = [F.interpolate(movement_embeding, size=(d,) + skip.shape[3:], mode=self.interpolation_mode) for skip in appearance_skips]
            skips = [torch.cat([a, b], dim=1) for a, b in zip(deformed_skips, kp_skips)]
        else:
            skips = deformed_skips

        video_deformed = self.deform_input(source_image, deformations_absolute)
        video_prediction = self.video_decoder(skips)
        video_prediction = self.refinement_module(video_prediction)
        # video_prediction = torch.sigmoid(video_prediction)

        return {"target_prediction": video_prediction, "target_deformed": video_deformed}


class PTPSModel:
    def __init__(self, opt):
        super(PTPSModel, self).__init__()
        self.name = 'PTPS'
        self.opt = opt
        self.gpu_ids = opt.gpu_ids if opt.gpu_ids else []
        config = yaml.load(open(os.path.join(opt.configure_file)))
        self.gener: PTPSGenerator = PTPSGenerator(**config['model_params']['generator_params'],
                                                  **config['model_params']['common_params'])
        self.discr: PTPSDiscriminator = PTPSDiscriminator(**config['model_params']['discriminator_params'],
                                                          **config['model_params']['common_params'])
        self.optimizer_G = torch.optim.Adam(self.gener.parameters(),
                                            lr=opt.lr,
                                            betas=(opt.beta1, 0.999))
        self.scheduler_G = get_scheduler(self.optimizer_G, opt)
        self.optimizer_D = torch.optim.Adam(self.discr.parameters(),
                                            lr=opt.lr,
                                            betas=(opt.beta1, 0.999))
        self.scheduler_D = get_scheduler(self.optimizer_D, opt)

        if self.opt.summary_writer:
            self.train_summary_writer = SummaryWriter(self.opt.train_summary)
            self.val_summary_writer = SummaryWriter(self.opt.val_summary)

    def cuda(self):
        self.gener.cuda()
        self.discr.cuda()

    def LRDecayStep(self, epoch):
        self.scheduler_G.step()
        self.scheduler_D.step()

    def train_batch(self, inputs: dict, loss: dict, metrics: dict, it):
        loss_accum = {}
        source_image, target_image, kp_driving, kp_source = \
            inputs["Source"], inputs["Target"], inputs["TargetKP"], inputs["SourceKP"]

        # training Generator
        fake_out = self.gener(source_image, kp_driving, kp_source)
        fake_prediction = fake_out["target_prediction"]
        fake_deformed = fake_out["target_deformed"]

        discriminator_maps_generated = self.discr(fake_prediction, kp_driving, kp_source)
        discriminator_maps_real = self.discr(target_image, kp_driving, kp_source)

        self.optimizer_G.zero_grad()

        loss_accum["loss_recon_deformed"] = loss["rec_deformed_loss"](discriminator_maps_real[0], fake_deformed)
        # loss_accum["loss_reconstruction"] = loss["rec_loss"](discriminator_maps_real, discriminator_maps_generated,
        #                                                      self.opt.rec_loss_weight)
        # loss_accum["loss_generator_gan"] = loss["generator_gan_loss"](discriminator_maps_generated)
        loss_bk = loss_accum["loss_recon_deformed"]# + loss_accum["loss_reconstruction"] +\
                  #loss_accum["loss_generator_gan"]

        loss_bk.backward(retain_graph=True)
        self.optimizer_G.step()

        # training Discriminator
        # discriminator_maps_generated = self.discr(fake_prediction, kp_driving, kp_source)
        # discriminator_maps_real = self.discr(source_image, kp_driving, kp_source)

        # self.optimizer_D.zero_grad()
        #
        # loss_accum["loss_discriminator_gan"] = loss["discriminator_gan_loss"](discriminator_maps_generated,
        #                                                                      discriminator_maps_real)
        # loss_bk = loss_accum["loss_discriminator_gan"]
        #
        # loss_bk.backward()
        # self.optimizer_D.step()
        #
        # if self.train_summary_writer:
        #     for x, y in list(loss_accum.items()):
        #         self.train_summary_writer.add_scalar(x, y.item(), it)

        make_vis(fake_out, inputs)
        return loss_accum

def make_vis(fake_out, inputs):
    source_image, target_image, kp_driving, kp_source = \
        inputs["Source"], inputs["Target"], inputs["TargetKP_OneHot"], inputs["SourceKP_OneHot"]

    deformed_source_image = fake_out["target_deformed"]

    [source_image, target_image, deformed_source_image] = \
        list(map(lambda x: x.squeeze(2), [source_image, target_image, deformed_source_image]))

    [kp_source, kp_driving] = list(map(lambda x: torch.sum(x, dim=1, keepdim=True), [kp_source, kp_driving]))

    [source_image, target_image, deformed_source_image, kp_driving, kp_source] = \
        list(map(lambda x: x.detach().cpu().numpy(), [source_image, target_image,
                                                      deformed_source_image,
                                                      kp_driving, kp_source]))

    # show the source, predicted target and ground truth target
    [src, trg, deform] = list(map(lambda x: (x.transpose([0, 2, 3, 1]) + 1) / 2.0 * 255.0,
                              [source_image, target_image, deformed_source_image]))
    img_cat = np.concatenate([src, trg, deform], 2)

    # show the pose result
    [src_kp, trg_kp] = list(map(lambda x: (x.transpose([0, 2, 3, 1])) * 255.,
                                        [kp_source, kp_driving]))
    kp_cat = np.concatenate([src_kp, trg_kp, trg_kp], 2)
    kp_cat = np.concatenate([kp_cat, ]*3, 3)

    # concat image and pose to one image
    total = np.concatenate([img_cat, kp_cat], 1)
    cv2.imwrite('test.png', total[0])

if __name__ == '__main__':
    pass