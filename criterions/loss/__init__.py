import torch.nn as nn
import torch.nn.functional as F
import torch

class IOULoss(nn.Module):
    '''
    IOU Loss: encourage warped feature to overlap target feature as well as possible
    '''
    def __init__(self):
        super(IOULoss, self).__init__()
        self.SMOOTH = 1e-6

    def iou(self, outputs, labels):
        intersection = (outputs * labels).sum((1, 2, 3))
        union = (torch.max(outputs, labels)).sum((1, 2, 3))
        iou = (intersection + self.SMOOTH) #/ (union + self.SMOOTH)
        return iou.mean()

    def forward(self, warped_parsing_pyrs, target_parsing_pyrs):
        res_loss = 0.
        for (warped_parsing, target_parsing) in zip(warped_parsing_pyrs, target_parsing_pyrs):
            res_loss += self.iou(warped_parsing, target_parsing)
        return res_loss

class NNLoss(nn.Module):
    '''
    Unofficial Implementation of Nearest Neighbor Loss in paper "Deformable GANs for Pose-based Human Image Generation"
    '''
    def __init__(self):
        super(NNLoss, self).__init__()
        self.neighbor = 5

    def forward(self, pred_target, gt_target):
        pad_gt_target = F.pad(gt_target, pad=[(self.neighbor-1)//2, (self.neighbor-1)//2, (self.neighbor-1)//2, (self.neighbor-1)//2, 0, 0, 0, 0], mode='constant', value=1e30*1.)
        stack_res = []
        for i in range(0, self.neighbor):
            for j in range(0, self.neighbor):
                if i == self.neighbor - 1 and j == self.neighbor - 1:
                    stack_res.append(pad_gt_target[:, :, i:, j:])
                elif i == self.neighbor - 1 and j != self.neighbor - 1:
                    stack_res.append(pad_gt_target[:, :, i:, j:-(self.neighbor-1-j)])
                elif i != self.neighbor - 1 and j == self.neighbor - 1:
                    stack_res.append(pad_gt_target[:, :, i:-(self.neighbor-1-i), j:])
                else:
                    stack_res.append(pad_gt_target[:, :, i:-(self.neighbor-1-i), j:-(self.neighbor-1-j)])
        stack_res = torch.stack(stack_res, dim=0)
        pred_target = pred_target.unsqueeze(0)

        diff_abs = torch.abs(stack_res - pred_target) #(neighbor, bs, c, h, w)
        sum_diff = torch.sum(diff_abs, dim=2) #(neighbor, bs, h, w)
        min_diff, _ = torch.min(sum_diff, dim=0)
        nn_loss = torch.sum(min_diff)

        return nn_loss

class RECLoss(nn.Module):
    def __init__(self):
        super(RECLoss, self).__init__()
        pass

    def forward(self, prediction, target):
        diff = torch.abs(prediction - target)
        return diff.view(diff.shape[0], -1).mean(-1).mean()

class RECONSLoss(nn.Module):
    def __init__(self):
        super(RECONSLoss, self).__init__()
        self.reconstruction_loss = RECLoss()

    def forward(self, discriminator_maps_real, discriminator_maps_generated, lamda):
        recons_loss = 0
        for i, (a, b) in enumerate(zip(discriminator_maps_real[:-1], discriminator_maps_generated[:-1])):
            if lamda[i] == 0:
                continue
            recons_loss += lamda[i] * self.reconstruction_loss(a, b)
        return recons_loss

class GENGANLoss(nn.Module):
    def __init__(self):
        super(GENGANLoss, self).__init__()
        pass

    def forward(self, discriminator_maps_generated):
        scores_generated = discriminator_maps_generated[-1]
        score = (1 - scores_generated) ** 2
        return score.view(score.shape[0], -1).mean(-1).mean()

class DISGANLoss(nn.Module):
    def __init__(self):
        super(DISGANLoss, self).__init__()
        pass

    def forward(self, discriminator_maps_generated, discriminator_maps_real):
        scores_real = discriminator_maps_real[-1]
        scores_generated = discriminator_maps_generated[-1]
        score = (1 - scores_real) ** 2 + scores_generated ** 2
        return score.view(score.shape[0], -1).mean(-1).mean()

loss_dict = {
    "MSE": nn.MSELoss(),
    "BCE": nn.BCELoss(),
    "L1" : nn.L1Loss(),
    "NNL": NNLoss(),
    "IOU": IOULoss(),
    "REC": RECLoss(),
    "RECONSLoss": RECONSLoss(),
    "GENGANLoss": GENGANLoss(),
    "DISGANLoss": DISGANLoss()
}

def create_loss_single(lamda, loss):
    return lambda *inputs: lamda * loss_dict[loss](*inputs)

def create_loss(opt):
    train_loss = opt.loss
    print(train_loss.items())
    loss = {}
    for k, v in train_loss.items():
        loss[k] = create_loss_single(*v)
    print('-> creating loss...')
    return loss
