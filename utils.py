import numpy as np
import os
from PIL import Image

def read_images(path, suffix=""):
    dirpath, dirnames, filenames = next(os.walk(path))
    fs = filter(lambda x: x.endswith(suffix), filenames)
    fs = sorted(fs)
    return np.array([Image.open(os.path.join(dirpath, f)) for f in fs])


def read_images_batch(path, batch_size, suffix=""):
    dirpath, dirnames, filenames = next(os.walk(path))
    fs = filter(lambda x: x.endswith(suffix), filenames)
    fs = sorted(fs)
    return np.array([Image.open(os.path.join(dirpath, fs[i])) for i in range(batch_size)])
