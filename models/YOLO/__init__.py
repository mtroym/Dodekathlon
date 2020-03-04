import torch


class YOLOModel:
    def __init__(self, opt):
        self.name = 'Creative Adversarial Network'
        self.opt = opt

        self.is_train = True
        self.batch_size = opt.batchSize
        self.gpu_ids = opt.gpu_ids if opt.gpu_ids else []
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and len(self.gpu_ids) > 0) else "cpu")
        self.dtype = torch.cuda.FloatTensor if self.device != torch.device("cpu") else torch.FloatTensor
        self.save_dir = opt.expr_dir
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        # self.schedular_D = get_scheduler(self.optimizer_D, opt)
        # self.schedular_G = get_scheduler(self.optimizer_G, opt)
        self.time_g = 10
        self.time_d = 10
        self.cuda()

    def cuda(self):
        pass

    def train_batch(self, inputs: dict, loss: dict, metrics: dict, niter: int = 0) -> dict:
        real = inputs["Source"].type(self.dtype)
        # real_label = inputs["Class"].type(self.dtype)
        # fake_label = torch.zeros((self.batch_size,), device=self.device).type(self.dtype)

        current_minibatch = real.shape[0]
        label = torch.full((current_minibatch,), 1).to(self.device)
        err_d_sum = 0
        self.optimizer_D.zero_grad()

        err_d = 0
        for _ in range(self.time_d):
            # train with real label for D
            random_noise = self.gen_random_noise(current_minibatch)
            fake = self.generator(random_noise)
            pred_real = self.discriminator(real)
            pred_fake = self.discriminator(fake)
            label.fill_(1)
            gradient_penalty = calculate_gradient_penatly(self.discriminator, real.data, fake.data, self.device)
            err_d_real = loss["bce_loss"](pred_real, label) + gradient_penalty
            err_d_real.backward(retain_graph=True)
            label.fill_(0)
            err_d_fake = loss["bce_loss"](pred_fake, label)
            err_d_fake.backward()
        self.optimizer_D.step()
        err_d_mean = err_d / self.time_d
        # for _ in range(self.time_g):

        self.optimizer_G.zero_grad()
        random_noise = self.gen_random_noise(current_minibatch)
        fake = self.generator(random_noise)
        pred_fake = self.discriminator(fake)
        label.fill_(0)
        err_g = loss["bce_loss"](pred_fake, label)
        # err_g = - torch.mean(pred_fake)
        err_g.backward()
        self.optimizer_G.step()

        with torch.no_grad():
            fake = self.generator(self.fixed_noise)

        return {
            "vis": {"Target": fake,
                    "Source": inputs["Source"]},
            "loss": {"Loss_G": err_g,
                     "Loss_D": err_d_mean,
                     # "D_fake": float(err_d_fake.mean().item()),
                     # "D_real": float(err_d_real.mean().item()),
                     # "Time_G": self.time_g
                     }
        }
