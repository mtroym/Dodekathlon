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
        outputs, labels = outputs.int(), labels.int()
        intersection = (outputs & labels).float().sum((1, 2, 3))
        union = (outputs | labels).float().sum((1, 2, 3))
        iou = (intersection + self.SMOOTH) / (union + self.SMOOTH)
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

loss_dict = {
    "MSE": nn.MSELoss(),
    "BCE": nn.BCELoss(),
    "L1" : nn.L1Loss(),
    "NNL": NNLoss(),
    "IOU": IOULoss(),
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
