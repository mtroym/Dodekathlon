import torch.utils.data as data


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
