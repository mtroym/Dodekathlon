# coding=utf-8
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
import torchvision.models as models


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class FeatureExtraction(nn.Module):
    def __init__(self, input_nc, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(FeatureExtraction, self).__init__()
        downconv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        model = [downconv, nn.ReLU(True), norm_layer(ngf)]
        for i in range(n_layers):
            in_ngf = 2 ** i * ngf if 2 ** i * ngf < 512 else 512
            out_ngf = 2 ** (i + 1) * ngf if 2 ** i * ngf < 512 else 512
            downconv = nn.Conv2d(in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model += [downconv, nn.ReLU(True)]
            model += [norm_layer(out_ngf)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]
        model += [norm_layer(512)]
        model += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.ReLU(True)]

        self.model = nn.Sequential(*model)
        init_weights(self.model, init_type='normal')

    def forward(self, x):
        return self.model(x)


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor


class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512, output_dim=6, use_cuda=False, scale_factor=1):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(1024 // 4 ** scale_factor, output_dim)
        self.tanh = nn.Tanh()
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()
            self.tanh.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        # print("x:shape:", x.size())
        x = self.linear(x)
        x = self.tanh(x)
        return x


class TpsGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.P_X_base = P_X.clone()
            self.P_Y_base = P_Y.clone()
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()
                self.P_X_base = self.P_X_base.cuda()
                self.P_Y_base = self.P_Y_base.cuda()

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta, torch.cat((self.grid_X, self.grid_Y), 3))

        return warped_grid

    def compute_L_inverse(self, X, Y):
        N = X.size()[0]  # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = torch.pow(Xmat - Xmat.transpose(0, 1), 2) + torch.pow(Ymat - Ymat.transpose(0, 1), 2)
        P_dist_squared[P_dist_squared == 0] = 1  # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        # construct matrix L
        O = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat((P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 1].size() + (1, self.N))

        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation - P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared == 0] = 1  # avoid NaN in log computation
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) + points_Y_batch.size()[1:])

        points_X_prime = A_X[:, :, :, :, 0] + \
                         torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = A_Y[:, :, :, :, 0] + \
                         torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        return torch.cat((points_X_prime, points_Y_prime), 3)


class GMM(nn.Module):
    """ Geometric Matching Module
    """

    def __init__(self, opt, scale_factor):
        super(GMM, self).__init__()
        self.opt = opt
        self.extractionA = FeatureExtraction(self.opt.keypoint + self.opt.semantic, ngf=64, n_layers=3,
                                             norm_layer=nn.BatchNorm2d)
        self.extractionB = FeatureExtraction(self.opt.keypoint + self.opt.semantic, ngf=64, n_layers=3,
                                             norm_layer=nn.BatchNorm2d)
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(input_nc=256 // 4 ** scale_factor, output_dim=2 * opt.grid_size ** 2, use_cuda=True, scale_factor=scale_factor)
        self.gridGen = TpsGridGen(256 // 2 ** scale_factor, 192 // 2 ** scale_factor, use_cuda=True, grid_size=opt.grid_size)

    def forward(self, inputA, inputB):
        featureA = self.extractionA(inputA)
        featureB = self.extractionB(inputB)
        featureA = self.l2norm(featureA)
        featureB = self.l2norm(featureB)
        # print('featureA size: ', featureA.size())
        # print('featureB size: ', featureB.size())
        correlation = self.correlation(featureA, featureB)
        # print('correlation size: ', correlation.size())
        theta = self.regression(correlation)
        # print('theta size: ', theta.size())
        grid = self.gridGen(theta)
        # print('grid_size: ', grid.size())
        return grid, theta


class SynthesisNet(nn.Module):
    def __init__(self, opt, pyramid_layer_nums=[19, 19, 19]):
        super(SynthesisNet, self).__init__()
        self.opt = opt

        self.feat_nums = [64, 64, 128, 256, 0, 0, 0, 0]
        self.conv_nets = []

        self.feat_nums[self.opt.pyramid_num - 1] = 0
        for i in range(self.opt.pyramid_num - 2, -1, -1):
            cur_module = nn.Sequential(
                nn.Conv2d(pyramid_layer_nums[i+1] + self.feat_nums[i+1], self.feat_nums[i], kernel_size=3, stride=1, padding=1),
                # nn.Conv2d(pyramid_layer_nums[i], self.feat_nums[i], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.feat_nums[i]),
                nn.ReLU(inplace=True))
            if len(opt.gpu_ids):
                cur_module = cur_module.cuda()
            self.conv_nets.append(cur_module)

        self.img_syn = nn.Sequential(nn.Conv2d(pyramid_layer_nums[0] + self.feat_nums[0], 3, kernel_size=3, stride=1, padding=1))

        if len(opt.gpu_ids):
            self.img_syn = self.img_syn.cuda()


    def forward(self, warped_pyrs):
        # res_pyrs = []
        last_f = warped_pyrs[-1]
        for f, net_module in zip(warped_pyrs[::-1][1:], self.conv_nets):
            # print("f size: ", f.size(), "last_f size: ", last_f.size())
            small_last_f = net_module(last_f)
            last_f = torch.cat([F.interpolate(small_last_f, scale_factor=2, mode='bilinear'), f], dim=1)
            # res_pyrs.append(last_f)
        res_img = self.img_syn(last_f)
        return res_img

class CTPSGenerator(nn.Module):
    '''
    CTPS: Controllable TPS
    '''

    def __init__(self, opt, dropout: float = None):
        super(CTPSGenerator, self).__init__()

        # set the require_grad property to false to freeze the vgg params
        self.vgg = models.vgg19(pretrained=True).features
        for p in self.parameters():
            p.requires_grad = False

        self.opt = opt
        self.dropout = dropout if dropout is not None else False
        self.keypoint = self.opt.keypoint
        self.semantic = self.opt.semantic

        scale_factor = 0
        self.gmm_pyrs = []
        for i in range(self.opt.pyramid_num):
            gmm = GMM(opt, scale_factor)
            if len(opt.gpu_ids):
                gmm = gmm.cuda()
            self.gmm_pyrs.append(gmm)
            scale_factor += 1

        self.synthesis_net = SynthesisNet(opt)
        if len(opt.gpu_ids):
            self.synthesis_net = self.synthesis_net.cuda()

    def feat_extract(self, img, layers=[3, 8, 13]):
        '''
        :param img:
        :param layers: [3, 8, 13]
        :return: returns a list including a raw img and its vgg pyramid features
        '''
        res = []

        # normalization for vgg features extraction
        source_norm = self.preprocess_img(img)

        # extract specific feature
        f = source_norm
        for i, layer in enumerate(list(self.vgg)):
            # print("vgg: ", i, layer)
            f = layer(f)
            if i in layers:
                # cat normalized raw img and its first pyramid feature
                if i == layers[0]:
                    res.append(torch.cat([source_norm, f], dim=1))
                else:
                    res.append(f)
            if len(res) >= self.opt.pyramid_num: break

        return res

    def preprocess_img(self, source):
        mean = torch.FloatTensor(3).to(source)
        mean[0] = 0.485
        mean[1] = 0.456
        mean[2] = 0.406
        mean = mean.resize(1, 3, 1, 1)

        std = torch.FloatTensor(3).to(source)
        std[0] = 0.229
        std[1] = 0.224
        std[2] = 0.225
        std = std.resize(1, 3, 1, 1)

        source_norm = (source + 1) / 2  # [-1, 1] => [0, 1]
        source_norm = (source_norm - mean) / std

        return source_norm

    def warp_feats(self, GMMNet, source_parsing, source_kp, target_parsing, target_kp, source):
        # warp image and vgg features based on computed TPS layer, remember to keep size same
        warped_sources = []
        for i in range(1, self.semantic):
            source_parsing_mask = torch.zeros_like(source_parsing).to(source_parsing)
            source_parsing_mask[:, i:i + 1, :, :] = 1
            target_parsing_mask = torch.zeros_like(target_parsing).to(target_parsing)
            target_parsing_mask[:, i:i + 1, :, :] = 1

            source_input_i = torch.cat([source_kp, source_parsing * source_parsing_mask], dim=1)
            target_input_i = torch.cat([target_kp, target_parsing * target_parsing_mask], dim=1)

            grid_i, theta_i = GMMNet(source_input_i, target_input_i)
            warped_source_i = F.grid_sample(source[:, i:i + 1, :, :] * source_parsing[:, i:i + 1, :, :], grid_i,
                                            padding_mode='border')
            warped_sources.append(warped_source_i)

        warped_sources_stack = torch.cat(warped_sources, dim=1)
        # warped_res, _ = torch.max(warped_sources_stack, dim=1, keepdim=True)

        return warped_sources_stack

    def forward(self, inputs: dict):
        source = inputs["Source"]
        source_kp = inputs["SourceKP"]
        target_kp = inputs["TargetKP"]
        source_parsing = inputs["SourceParsing"]
        target_parsing = inputs["TargetParsing"]

        if len(self.opt.gpu_ids):
            source, source_kp, target_kp, source_parsing, target_parsing = \
                list(map(lambda x: x.cuda(), [source, source_kp, target_kp, source_parsing, target_parsing]))

        # build kp and ps pyramids
        source_parsing_pyrs, source_kp_pyrs, target_parsing_pyrs, target_kp_pyrs = \
            [source_parsing, ], [source_kp, ], [target_parsing, ], [target_kp, ]

        tmp_sp, tmp_sk, tmp_tp, tmp_tk = source_parsing, source_kp, target_parsing, target_kp
        for _ in range(self.opt.pyramid_num - 1):
            tmp_sp, tmp_sk, tmp_tp, tmp_tk = list(map(lambda x: F.interpolate(x, scale_factor=0.5, mode='nearest'),
                                                      [tmp_sp, tmp_sk, tmp_tp, tmp_tk]))

            source_parsing_pyrs.append(tmp_sp)
            source_kp_pyrs.append(tmp_sk)
            target_parsing_pyrs.append(tmp_tp)
            target_kp_pyrs.append(tmp_tk)
        # print("len target_parsing_pyrs: ", len(target_parsing_pyrs))
        # print("len target_kp_pyrs: ", len(target_kp_pyrs))
        # feature extraction
        feat_pyrs = self.feat_extract(source)
        # print("len feat_pyrs: ", len(feat_pyrs))

        # construct warped pyramids based on learned TPS pyramids
        warped_pyrs = []
        for i in range(self.opt.pyramid_num):
            warped_src = self.warp_feats(self.gmm_pyrs[i], source_parsing_pyrs[i], source_kp_pyrs[i],
                                         target_parsing_pyrs[i], target_kp_pyrs[i], feat_pyrs[i])
            warped_pyrs.append(warped_src)
            # print('feat_pyr size: ', feat_pyrs[i].size())
            # print('warped_src size: ', warped_src.size())
        # synthesize target image
        pred_trg = self.synthesis_net(warped_pyrs)

        return warped_pyrs, pred_trg


class CTPSModel:
    def __init__(self, opt):
        self.opt = opt
        self.gener: CTPSGenerator = CTPSGenerator(opt)
        self.optimizer_G = [torch.optim.Adam(filter(lambda p: p.requires_grad, \
                                                    self.gener.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999),
                                             eps=1e-08, weight_decay=1e-5)]

    def train_batch(self, inputs: dict, loss: dict, metrics: dict) -> dict:
        pass


if __name__ == '__main__':
    gmm = GMM()
