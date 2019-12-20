import torch.nn as nn

loss_dict = {
    "MSE": nn.MSELoss(),
    "BCE": nn.BCELoss(),
    "L1" : nn.L1Loss()
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
