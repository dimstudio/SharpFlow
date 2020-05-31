from torch.utils.data import Dataset
import numpy as np


class transportation_dataset(Dataset):
    def __init__(self, data_path):
        super(self)
        # Do something with the data_path
        # e.g. load it in memory
        self.dataset = np.random.random((42, 13, 37))

    def __getitem__(self, item):
        # usually input and target come from the same dataset. These are just dummy values
        values = self.dataset[item]
        target = np.random.randint(0, 8)
        return values, target

    def __len__(self):
        return len(self.dataset)