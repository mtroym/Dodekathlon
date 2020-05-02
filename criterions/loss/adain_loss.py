import torch
import torch.nn.functional as F


def content_loss(source: torch.Tensor, target: torch.Tensor):
    # assert not target.requires_grad
    return F.mse_loss(source, target).mean()


def style_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # assert not target.requires_grad
    assert target.shape[:2] == source.shape[:2]
    b, c, _, _ = target.shape
    target_view = target.view((b, c, -1))
    source_view = source.view((b, c, -1))
    src_mean, src_std = source_view.mean(-1), source_view.std(-1)
    trg_mean, trg_std = target_view.mean(-1), target_view.std(-1)
    return F.mse_loss(src_mean, trg_mean).mean() + F.mse_loss(src_std, trg_std).mean()


def style_loss_dict(source: dict, target: dict):
    union_keys = set(source.keys()).intersection(set(target.keys()))
    loss = [style_loss(source[key], target[key]).mean() for key in union_keys]
    # print(loss)
    l_val = loss[0]
    if len(loss) > 1:
        for l in loss[1:]:
            l_val += l
    return l_val
