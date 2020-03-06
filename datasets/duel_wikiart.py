import random

from datasets.wikiart import ArtsDataset


class DuelDataset:
    def __init__(self, opt):
        self.art = ArtsDataset(opt)

    def __getitem__(self, item):
        art_entry = self.art[(item % len(self))]
        art_entry_1 = self.art[(item + random.randint(0, 100)) % len(self)]
        return {
            "Source": art_entry["Source"],
            "Style": art_entry_1["Source"]
        }

    def __len__(self):
        return len(self.art)
