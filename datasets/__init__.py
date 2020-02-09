from torch.utils.data import DataLoader

from datasets.base import BaseDataset
from datasets.close_btn import CloseButton
from datasets.keypoint import KeypointDataset
from datasets.keypoint_parsing import KeypointParsingDataset
from datasets.wikiart import ArtsDataset


def create_dataset(opt):
    dataset = None
    if opt.dataset == 'Base':
        dataset = BaseDataset()
    elif opt.dataset in ["deepfashion256", "market1501", "deepfashion512"]:
        dataset = KeypointDataset(opt)
    elif opt.dataset in ["wikiart", "painting", "Rothko"]:
        dataset = ArtsDataset(opt)
    elif opt.dataset in ["close_btn"]:
        dataset = CloseButton(opt)
    return dataset


def create_dataloader(opt):
    print('=> creating dataloader...')
    dataset = create_dataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batchSize,
                            shuffle=not opt.serial_batches,
                            num_workers=int(opt.num_workers))
    return dataloader
