import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    pad_layer = nn.ZeroPad2d(padding=(-1, 0, 0, 0))
    input = torch.randn(1, 1, 3, 3)
    print(pad_layer(input).shape)


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
        self.crop_l = nn.ZeroPad2d(padding=(-1, 0, 0, 0))
        self.crop_r = nn.ZeroPad2d(padding=(0, -1, 0, 0))
        self.crop_t = nn.ZeroPad2d(padding=(0, 0, -1, 0))
        self.crop_b = nn.ZeroPad2d(padding=(0, 0, 0, -1))

    def forward(self, inputs):
        diff_lr = F.mse_loss(self.crop_l(inputs), self.crop_r(inputs))
        diff_tb = F.mse_loss(self.crop_t(inputs), self.crop_b(inputs))
        return diff_lr.mean() + diff_tb.mean()
