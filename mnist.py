import numpy as np
import os

DATA_DIR = "/data/ziz/not-backed-up/datasets-ziz-only"

def load_data(which_set="train", mode="all"):
    assert which_set in ['train', 'valid', 'test'], \
            "which_set takes values in [train, valid, test]"
    assert mode in ['all', 'batch'], \
            "mode takes values in [all, batch]"
    if mode == 'all':
        path = os.path.join(DATA_DIR, "raw_data/mnist", "mnist-{0}.npz".format(which_set))
        return load_data_all(path)
    elif mode == 'batch':
        return load_data_batch()

def load_data_all(path):
    data = np.load(path)
    return data

def load_data_batch():
    pass
