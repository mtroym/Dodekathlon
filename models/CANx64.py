from torch import nn

nz = 100  # Size of z latent vector (i.e. size of generator input)
ngf = 64  # Size of feature maps in generator
nc = 3
ndf = 64


class Generator(nn.Module):
    def __init__(self, ngpu, u="trans"):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.trans = u == "trans"
        layers = []
        # x2
        if self.trans:
            layers += [nn.ConvTranspose2d(nz, ngf * 8, 4, 2, 1, bias=False)]
        else:
            layers += [nn.UpsamplingBilinear2d(scale_factor=2),
                       nn.Conv2d(nz, ngf * 8, 3, 1, 1, bias=False)]
        layers += [nn.BatchNorm2d(ngf * 8), nn.ReLU(True)]

        # x4
        if self.trans:
            layers += [nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)]
        else:
            layers += [nn.UpsamplingBilinear2d(scale_factor=2),
                       nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1, bias=False)]
        layers += [nn.BatchNorm2d(ngf * 4), nn.ReLU(True)]

        # x8
        if self.trans:
            layers += [nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)]
        else:
            layers += [nn.UpsamplingBilinear2d(scale_factor=2),
                       nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False)]
        layers += [nn.BatchNorm2d(ngf * 2), nn.ReLU(True)]

        # x16
        if self.trans:
            layers += [nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)]
        else:
            layers += [nn.UpsamplingBilinear2d(scale_factor=2),
                       nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False)]
        layers += [nn.BatchNorm2d(ngf), nn.ReLU(True)]

        # x32
        if self.trans:
            layers += [nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)]
        else:
            layers += [nn.UpsamplingBilinear2d(scale_factor=2),
                       nn.Conv2d(ngf, nc, 3, 1, 1, bias=False)]
        layers += [nn.Tanh()]

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        # self.norm = SpectralNorm if True else nn.BatchNorm2d()
        # self.ngpu = ngpu
        # self.conv1 = SpectralNorm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        # self.conv2 = SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        # self.conv3 = SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        # self.conv4 = SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))
        # self.conv5 = SpectralNorm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))
        # self.leak = 0.2

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        """
        random_noise = torch.rand_like(input_data, requires_grad=False) * 0.001
        x = self.conv1(input_data + random_noise)
        x = nn.LeakyReLU(self.leak)(self.conv2(x))
        x = nn.LeakyReLU(self.leak)(self.conv3(x))
        x = nn.LeakyReLU(self.leak)(self.conv4(x))
        x = nn.LeakyReLU(self.leak)(self.conv5(x))
        pred = torch.sigmoid(x)
        """
        pred = self.main(input_data)
        return pred.flatten()
