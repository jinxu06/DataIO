import numpy as np
import os
import pprint

from data_provider import DataProvider

DATA_DIR = "/data/ziz/not-backed-up/datasets-ziz-only"
import pprint

def info():
    meta_dict = {
        "name": "mnist",
        "location": os.path.join(DATA_DIR, "raw_data/mnist", "mnist.npz"),
        "available set": ['train', 'test'],
        "available mode": ['all', 'batch'],
    }
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(meta_dict)

def load_data(which_set="train", mode="all"):
    assert which_set in ['train', 'valid', 'test'], \
            "which_set takes values in [train, test]"
    assert mode in ['all', 'batch'], \
            "mode takes values in [all, batch]"
    if mode == 'all':
        path = os.path.join(DATA_DIR, "raw_data/mnist", "mnist.npz")
        return _load_data_all(path, which_set)
    elif mode == 'batch':
        return _load_data_batch()

def _load_data_all(path, which_set):
    data = np.load(path)
    return data["x_"+which_set], data['y_'+which_set]

def _load_data_batch(path, which_set, batch_size):
    pass

class MNISTIterator(DataProvider):

    def __init__(self, which_set="train", batch_size=100, max_num_batches=-1, shuffle_order=True):

        inputs, targets = load_data(which_set, "all")
        super().__init__(
            inputs, targets, batch_size, max_num_batches, shuffle_order, rng=None)

    def next(self):
        """Returns next data batch or raises `StopIteration` if at end."""
        inputs_batch, targets_batch = super(MNISTDataProvider, self).next()
        return inputs_batch, targets_batch

    
