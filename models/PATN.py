# -*- coding: utf-8 -*-
"""
@Project:   pytorch-train
@File   :   PATN
@Author :   TonyMao@AILab
@Date   :   2019/11/26
@Desc   :   None
"""
import functools
import os

import torch
import torch.nn as nn

from .blocks import ResnetDiscriminator
from .helpers import custom_pad, get_norm_layer, get_scheduler, init_weights


class PATNDiscriminator(nn.Module):
    def __init__(self, opt, in_channel, hidden, backbone="resnet",
                 num_blocks=3, norm="instance", use_sigmoid=False,
                 init_type='normal', gpu_ids: list = None,
                 dropout=0.5, n_downsampling=2):
        super(PATNDiscriminator, self).__init__()
        self.opt = opt
        self.gpu_ids = gpu_ids if gpu_ids else []
        self.norm_layer = get_norm_layer(norm)
        self.dropout = dropout if dropout is not None else False
        self.discriminators = []
        if backbone == "resnet":
            self.core_fake_kp = ResnetDiscriminator(in_channel * 2, hidden, norm_layer=self.norm_layer,
                                                    use_dropout=dropout, use_sigmoid=use_sigmoid, n_blocks=num_blocks,
                                                    padding_type='reflect', n_downsampling=n_downsampling)

            self.core_fake_target = ResnetDiscriminator(in_channel + self.opt.keypoint, hidden, norm_layer=self.norm_layer,
                                                        use_dropout=dropout, use_sigmoid=use_sigmoid, n_blocks=num_blocks,
                                                        padding_type='reflect', n_downsampling=n_downsampling)
            self.discriminators = [self.core_fake_kp, self.core_fake_target]
        else:
            raise NotImplementedError("model backbone not implemented.")
        if len(self.gpu_ids) != 0:
            self.core_fake_kp.cuda()
            self.core_fake_target.cuda()
        if not self.opt.resume:
            init_weights(self.core_fake_kp, init_type)
            init_weights(self.core_fake_target, init_type)

    def forward(self, inputs: dict):
        outputs = {}
        source = inputs["FakeImage"]
        keypoint = inputs["TargetKP"]
        target = inputs["Target"]
        assert source.shape[2:] == keypoint[2:]
        outputs["PredFakePose"] = self.core_fake_kp(torch.cat([source, keypoint], dim=1))
        outputs["PredFakeAppearance"] = self.core_fake_target(torch.cat([source, target], dim=1))
        return outputs


class PATNEncoder(nn.Module):
    def __init__(self, in_channel, hidden, norm_layer=nn.BatchNorm2d, n_downsampling=2):
        super(PATNEncoder, self).__init__()
        self.in_channel = in_channel
        self.hidden = hidden
        self.activation = nn.ReLU
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.model_list = [
            nn.ReflectionPad2d(3),
            *make_encoder_layer(in_channel=self.in_channel, hidden=self.hidden,
                                norm_layer=norm_layer, kernel_size=7, padding=0,
                                use_bias=use_bias, activation=self.activation),
        ]
        for i in range(n_downsampling):
            in_cha = self.hidden * 2 ** i
            out_cha = self.hidden * 2 ** (i + 1)
            self.model_list += make_encoder_layer(in_channel=in_cha, hidden=out_cha, norm_layer=norm_layer,
                                                  kernel_size=3, stride=2, padding=1, use_bias=use_bias,
                                                  activation=self.activation)
        self.model = nn.Sequential(*self.model_list)
        self.model.float()

    def forward(self, inputs):
        return self.model(inputs)


def make_encoder_layer(in_channel, hidden, norm_layer, kernel_size=7, padding=0, use_bias=True, activation=nn.ReLU, stride=1):
    return [nn.Conv2d(in_channel, hidden, kernel_size=kernel_size,
                      padding=padding, bias=use_bias, stride=stride),
            norm_layer(hidden),
            activation(inplace=True)]


class PATNDecoder(nn.Module):
    def __init__(self, out_channel, hidden, norm_layer=nn.BatchNorm2d, n_downsampling=2):
        super(PATNDecoder, self).__init__()
        self.hidden = hidden
        self.activation = nn.ReLU
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.model_list = []
        for i in range(n_downsampling):
            in_cha = self.hidden * 2 ** (n_downsampling - i)
            out_cha = self.hidden * 2 ** (n_downsampling - i - 1)
            self.model_list += make_decoder_layer(in_channel=in_cha, hidden=out_cha, norm_layer=norm_layer,
                                                  kernel_size=3, stride=2, padding=1, use_bias=use_bias,
                                                  activation=self.activation)
        self.model_list += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=hidden, out_channels=out_channel, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*self.model_list)

    def forward(self, inputs):
        return self.model(inputs)


def make_decoder_layer(in_channel, hidden, norm_layer, kernel_size=7, padding=0, use_bias=True, activation=nn.ReLU, stride=1):
    return [nn.ConvTranspose2d(in_channel, hidden, kernel_size=kernel_size,
                               padding=padding, bias=use_bias, stride=stride),
            norm_layer(hidden),
            activation(inplace=True)]


def build_pathway_block(in_channel, padding_type, norm_layer, use_dropout, use_bias, cat_with_s=False, cal_att=False):
    conv_block = []
    conv_block, p = custom_pad(conv_block, padding_type)

    if cat_with_s:
        conv_block += [nn.Conv2d(in_channel * 2, in_channel * 2, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(in_channel * 2),
                       nn.ReLU(True)]
    else:
        conv_block += [nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(in_channel),
                       nn.ReLU(True)]
    if use_dropout:
        conv_block += [nn.Dropout(use_dropout)]
    conv_block, p = custom_pad(conv_block, padding_type)

    if cal_att:
        if cat_with_s:
            conv_block += [nn.Conv2d(in_channel * 2, in_channel, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=p, bias=use_bias)]
    else:
        conv_block += [nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(in_channel)]

    return nn.Sequential(*conv_block)


class PATBlock(nn.Module):
    def __init__(self, in_channel, padding_type, norm_layer, use_dropout: float = None, use_bias=True, cat_with_s=False):
        super(PATBlock, self).__init__()
        self.conv_p = build_pathway_block(in_channel, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        self.conv_s = build_pathway_block(in_channel, padding_type, norm_layer, use_dropout, use_bias, cal_att=True, cat_with_s=cat_with_s)

    def forward(self, f_tp, f_ts):
        p_out_inter = self.conv_p(f_tp)  # image path way.
        s_out_inter = self.conv_s(f_ts)  # pose path way.
        mask = torch.sigmoid(s_out_inter)  # attention.

        masked_p_out = p_out_inter * mask
        p_out = f_tp + masked_p_out  # residual connection

        # stream2 receive feedback from stream1
        s_out = torch.cat((s_out_inter, p_out), 1)
        return p_out, s_out, masked_p_out


class PATNGenerator(nn.Module):
    def __init__(self, opt, out_channel, hidden, backbone="PATN", num_blocks=3, norm="instance", use_sigmoid=False,
                 init_type='normal', padding_type='reflect', gpu_ids: list = None, dropout: float = None, n_downsampling=2):
        self.opt = opt
        super(PATNGenerator, self).__init__()
        self.dropout = dropout if not dropout else self.opt.dropout_rate
        self.keypoint = self.opt.keypoint
        self.gpu_ids = [] if gpu_ids is None else gpu_ids
        self.norm_layer = get_norm_layer(norm)
        self.num_blocks = num_blocks
        self.dropout = dropout if dropout is not None else False
        self.encoder_poses = PATNEncoder(self.keypoint * 2, hidden, norm_layer=self.norm_layer, n_downsampling=n_downsampling)
        self.encoder_image = PATNEncoder(out_channel, hidden, norm_layer=self.norm_layer, n_downsampling=n_downsampling)
        self.decoder = PATNDecoder(out_channel, hidden, norm_layer=self.norm_layer, n_downsampling=n_downsampling)
        self.transfer = []
        if type(self.norm_layer) == functools.partial:
            use_bias = self.norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = self.norm_layer == nn.InstanceNorm2d
        if backbone == "PATN":
            for i in range(num_blocks):
                self.transfer.append(PATBlock(hidden * 2 ** n_downsampling, padding_type=padding_type, norm_layer=self.norm_layer,
                                              use_dropout=dropout, use_bias=use_bias,
                                              cat_with_s=False if i == 0 else True))
        else:
            raise NotImplementedError("generator backbone not implemented.")

    def forward(self, inputs: dict):
        source = inputs["Source"]
        source_kp = inputs["SourceKP"]
        target_kp = inputs["TargetKP"]

        source_encoded = self.encoder_image(source)
        keypoint_encoded = self.encoder_poses(torch.cat([source_kp, target_kp], dim=1))

        for i in range(self.num_blocks):
            source_encoded, keypoint_encoded, _ = self.transfer[i](source_encoded, keypoint_encoded)

        decoded = self.decoder(source_encoded)
        return {"FakeImage": decoded}


class PATNTransferModel:
    def __init__(self, opt):
        self.name = 'PATN'
        self.opt = opt
        self.is_train = True
        self.batch_size = opt.batchSize
        self.gpu_ids = opt.gpu_ids if opt.gpu_ids else []
        self.dtype = torch.cuda.FloatTensor if len(self.gpu_ids) != 0 else torch.FloatTensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.discr: PATNDiscriminator = PATNDiscriminator(opt, in_channel=self.opt.in_channel, hidden=self.opt.hidden)
        self.gener: PATNGenerator = PATNGenerator(opt, out_channel=self.opt.in_channel, hidden=self.opt.hidden)
        self.optimizers_D = [torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                             for net in self.discr.discriminators]
        self.optimizer_G = torch.optim.Adam(self.gener.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.schedulars_D = [get_scheduler(optimizer, opt) for optimizer in self.optimizers_D]
        self.schedular_G = get_scheduler(self.optimizer_G, opt)

    def cuda(self):
        self.discr.cuda()
        self.gener.cuda()
        # [optim.cuda() for optim in self.optimizers_D]
        # self.optimizer_G.cuda()

    def train_batch(self, inputs: dict, loss: dict, metrics: dict) -> dict:
        fake_out = self.gener(inputs)
        # G once.
        self.optimizer_G.zero_grad()
        pred_out = self.discr(fake_out.update(inputs))
        loss_gan = loss["GANLoss"](pred_out["fake"], 0)
        loss_gan += loss["GANLoss"](pred_out["real"], 1)
        loss_gan.backword()
        # note.
        self.optimizer_G.step()
        for i in range(len(self.optimizers_D)):
            self.optimizers_D[i].zero_grad()

        # loss_G = self.critierion(out, False/True)
        # loss#
        out = self.gener(inputs)
        return {"visuals": None, "scalars": None}

    def eval_batch(self, inputs: dict, loss: dict = None, metrics: dict = None):
        return {"visuals": None, "scalars": None}


if __name__ == '__main__':
    inputs_data = {}
    inputs_data['Source'] = torch.zeros((2, 3, 256, 256))
    inputs_data['SourceKP'] = torch.zeros((2, 18, 256, 256))
    inputs_data['Target'] = torch.zeros((2, 3, 256, 256))
    inputs_data['TargetKP'] = torch.zeros((2, 18, 256, 256))
