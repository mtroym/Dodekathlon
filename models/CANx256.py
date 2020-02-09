import torch
import torch.nn as nn


class CANGenerator(nn.Module):
    def __init__(self, latent_dim, hidden, norm_layer=nn.BatchNorm2d, config=None):
        super(CANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.activation = nn.ReLU
        if config is None:
            self.size_list = [4, 8, 16, 32, 64, 128, 256]
            self.channel_list = [hidden, hidden / 2, hidden / 4, hidden / 8, hidden / 8, hidden / 16, 3]
        else:
            self.size_list = config["size_list"]
            self.channel_list = config["channel_list"]

        # input to the network
        init_size = self.size_list[0]
        # self.header = nn.Linear(self.latent_dim, init_size * init_size * self.hidden, bias=False)  # BN x 4 x 4 x 1024
        self.header = nn.ConvTranspose2d(self.latent_dim, out_channels=self.hidden,
                                         kernel_size=init_size, bias=False)  # BN x 4 x 4 x 1024
        self.norm = norm_layer(self.hidden, eps=1e-5, momentum=0.9)
        self.model_list = []
        for i in range(len(self.channel_list) - 1):
            in_cha = int(self.channel_list[i])
            out_cha = int(self.channel_list[i + 1])
            # print(in_cha, out_cha)
            self.model_list.append(nn.ConvTranspose2d(in_cha, out_cha, kernel_size=4, stride=2, padding=1, padding_mode="zeros", bias=False))
            self.model_list.append(norm_layer(out_cha, eps=1e-5, momentum=0.9))
            if i != len(self.channel_list) - 2:
                self.model_list.append(self.activation(inplace=True))
            else:
                self.model_list.append(nn.Tanh())
        self.core = nn.Sequential(
            *self.model_list
        )

        self.norm.float()
        self.header.float()
        self.core.float()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # batch_size = inputs.shape[0]
        head = self.header(inputs)
        # print(linear.shape)
        # linear_reshape = linear.reshape((batch_size, self.hidden, self.size_list[0], self.size_list[0]))
        normalized = self.norm(head)
        # print(normalized.shape)
        activated = self.activation(inplace=True)(normalized)
        fake_image = self.core(activated)
        return fake_image


class CANDiscriminator(nn.Module):
    def __init__(self, in_channel=3, num_class=2, norm_layer=nn.BatchNorm2d, config=None):
        super(CANDiscriminator, self).__init__()
        self.activation = nn.LeakyReLU
        self.channel_list = [32, 64, 128, 256, 512, 512]
        self.norm = norm_layer
        self.in_channel = in_channel
        self.model_list = []
        self.enter = nn.Conv2d(self.in_channel, self.channel_list[0], kernel_size=4, stride=2, padding=1, padding_mode="reflex", bias=False)
        self.enter_act = self.activation(0.2)
        for i in range(len(self.channel_list) - 1):
            self.model_list.append(nn.Conv2d(self.channel_list[i], self.channel_list[i + 1],
                                             kernel_size=4, stride=2, padding=1, padding_mode="reflex", bias=False))
            self.model_list.append(self.norm(self.channel_list[i + 1]))
            self.model_list.append(self.activation(0.2, inplace=True))
        self.core = nn.Sequential(*self.model_list)
        self.classifier_rf = nn.Sequential(
            nn.Conv2d(self.channel_list[-1], 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # random_noise = torch.rand_like(inputs, requires_grad=False) * 0.001
        before_core = self.enter_act(self.enter(inputs))
        score = self.core(before_core)
        # print("Score", score.shape)
        discriminator_output = self.classifier_rf(score)
        return discriminator_output.flatten()
