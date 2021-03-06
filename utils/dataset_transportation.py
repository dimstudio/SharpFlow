from torch.utils.data import Dataset
import numpy as np
import pickle
import os
import torch


class transportation_dataset(Dataset):
    def __init__(self, data_path, train=True, use_magnitude=False, use_lstm=False):
        # super(self)
        # Do something with the data_path
        # e.g. load it in memory
        if train:
            data_path = os.path.join(data_path, "train")
        else:
            data_path = os.path.join(data_path, "test")

        with open(os.path.join(data_path, "sensor_data.pkl"), "rb") as f:
            self.data = pickle.load(f)
        with open(os.path.join(data_path, "annotations.pkl"), "rb") as f:
            self.targets = pickle.load(f)

        self.use_magnitude = use_magnitude
        if not use_magnitude and not use_lstm:
            # samples x seq_len x features --> samples x features x seq_len
            self.data = np.transpose(self.data, (0, 2, 1))

        self.data = torch.tensor(self.data, dtype=torch.float)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __getitem__(self, item):
        values = self.data[item]
        target = self.targets[item]
        if self.use_magnitude:
            values = np.expand_dims(values, 0)
        return values, target

    def __len__(self):
        return len(self.targets)