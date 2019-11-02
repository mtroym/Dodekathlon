from .loss import base as loss_base
from .metrics import base as metrics_base


def create_criterion(opt):
    print("=> creating criterions...")
    loss = loss_base.create_loss(opt)
    metrics = metrics_base.create_metrics(opt)
    return loss, metrics
