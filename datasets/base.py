import torch
import torch.utils.data as data

from datasets.keypoint import KeypointDataset


def create_dataloader(opt):
    print('=> creating dataloader...')
    dataset = create_dataset(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads))
    return dataloader


def create_dataset(opt):
    dataset = BaseDataset()
    if opt.dataset == 'Base':
        pass
    elif opt.dataset == "KP":
        dataset = KeypointDataset()
    return dataset


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        return 1000

    def __getitem__(self, item):
        return None

    @property
    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass
