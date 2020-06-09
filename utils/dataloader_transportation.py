from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
from utils.dataset_transportation import transportation_dataset


def get_train_valid_loader(data_dir, batch_size, valid_size, shuffle=True, num_workers=0, pin_memory=False):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    train_dataset = transportation_dataset(data_path=data_dir, train=True)
    valid_dataset = transportation_dataset(data_path=data_dir, train=True)

    # Split the data into training and validation set
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, valid_loader


def get_test_loader(data_dir, batch_size, num_workers=0, pin_memory=False):
    test_dataset = transportation_dataset(data_path=data_dir, train=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return test_loader
