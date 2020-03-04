import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Implements adaptive instance normalization (AdaIN) as described in the paper:
Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization
Xun Huang, Serge Belongie

based on `https://github.com/xunhuang1995/AdaIN-style/blob/master/lib/AdaptiveInstanceNormalization.lua`
"""


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, content, style: torch.Tensor = None):
        b, c, Hc, Wc = content.shape
        assert c == self.num_features, "the input channel is not same with feature"
        assert style is not None or (self.weight is not None and self.bias is not None)

        use_input_state = False if style is None else True
        if use_input_state:
            ns, _, Hs, Ws = style.shape
            assert ns == b, "style must has only same"
            style_reshape = style.contiguous().view(ns, c, Hs * Ws)
            self.set_weight_bias(weight=style_reshape.std(2, True).view(-1),
                                 bias=style_reshape.mean(2).view(-1))

        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        content_reshaped = content.contiguous().view(
            1, b * c, Hc, Wc)

        # Apply instance norm
        out = F.batch_norm(
            content_reshaped, running_mean, running_var, self.weight,
            self.bias, True, self.momentum, self.eps)

        # Init.
        self.weight = None
        self.bias = None
        return out.view_as(content)

    def set_weight_bias(self, weight, bias):
        assert weight.shape == bias.shape
        self.weight = weight
        self.bias = bias

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


if __name__ == '__main__':
    adain = AdaptiveInstanceNorm2d(num_features=256)
    content = torch.randn(
        10, 256, 32, 32, requires_grad=True
    )
    style = torch.randn(10, 256, 16, 16, requires_grad=True)
    a = adain(content, style)
    print(content.shape, a.shape)
