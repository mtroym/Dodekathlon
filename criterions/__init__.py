from criterions.loss import create_loss
from criterions.metrics import create_metrics


def create_criterion(opt):
    print("=> creating criterions...")
    loss = create_loss(opt)
    metrics = create_metrics(opt)
    return loss, metrics
