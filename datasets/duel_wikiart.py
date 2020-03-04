from datasets.close_btn import CloseButton
from datasets.wikiart import ArtsDataset


class DuelDataset:
    def __init__(self, opt):
        self.art = ArtsDataset(opt)
        self.content = CloseButton(opt)

    def __getitem__(self, item):
        artentry = self.art[item]
        contententry = self.content[item]

    def __len__(self):
        return 100000
