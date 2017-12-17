import numpy as np
import os
from PIL import Image

def read_images(path, suffix=""):
    dirpath, dirnames, filenames = next(os.walk(path))
    fs = filter(lambda x: x.endswith(suffix), filenames)
    fs = sorted(fs)
    return np.array([np.array(Image.open(os.path.join(dirpath, f))) for f in fs])


def read_images_batch(path, batch_size, suffix=""):
    dirpath, dirnames, filenames = next(os.walk(path))
    fs = filter(lambda x: x.endswith(suffix), filenames)
    fs = sorted(fs)
    return np.array([np.array(Image.open(os.path.join(dirpath, fs[i]))) for i in range(batch_size)])


def inspect_data_dirs(dirs):
    print("Inspect data folders......")
    for d in dirs:
        if os.path.exists(d):
            dirpath, dirnames, filenames = next(os.walk(d))
            if len(dirnames)+len(filenames) ==0:
                print("{0}: folder exists but empty".format(d))
            else:
                print("{0}: {1} directories, {2} files".format(d, len(dirnames), len(filenames)))
        else:
            print("{0}: folder not found or not accessible".format(d))
