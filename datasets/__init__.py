from torch.utils.data import DataLoader

from datasets.base import BaseDataset
from datasets.close_btn import CloseButton
from datasets.duel_wikiart import DuelDataset
from datasets.keypoint import KeypointDataset
from datasets.keypoint_parsing import KeypointParsingDataset
from datasets.wikiart import ArtsDataset


def create_dataset(ds, opt):
    dataset = None
    if ds == 'Base':
        dataset = BaseDataset()
    elif ds in ["deepfashion256", "market1501", "deepfashion512"]:
        dataset = KeypointDataset(opt)
    elif ds in ["wikiart", "painting", "Rothko"]:
        dataset = ArtsDataset(opt)
    elif ds in ["close_btn"]:
        dataset = CloseButton(opt)
    elif ds in ["duel"]:
        dataset = DuelDataset(opt)
    return dataset


def create_dataloader(opt):
    print('=> creating dataloader...')
    if isinstance(opt.dataset, list):
        raise NotImplementedError("multi-dataset not implemented.")
        dataloader_dict = {}
        for ds in opt.dataset:
            dataset = create_dataset(ds, opt)
            dataloader_dict[ds] = DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=not opt.serial_batches,
                                             num_workers=int(opt.num_workers))
        return dataloader_dict

    else:
        dataset = create_dataset(opt.dataset, opt)
        dataloader = DataLoader(dataset, batch_size=opt.batchSize,
                                shuffle=not opt.serial_batches,
                                num_workers=int(opt.num_workers))
        return dataloader
