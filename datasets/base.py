import torch.utils.data as data
from PIL import Image

# from datasets.close_btn import CloseButton
# from datasets.keypoint import KeypointDataset
# from datasets.keypoint_parsing import KeypointParsingDataset
# from datasets.wikiart import ArtsDataset
Image.MAX_IMAGE_PIXELS = 1000000000


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
