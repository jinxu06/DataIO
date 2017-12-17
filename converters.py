# http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
import numpy as np
import tables
import os
from PIL import Image


def convert_images_dir_to_h5(input_dir, output_name, img_type=None, img_shape=None, suffix=None):
    if not os.path.exists(input_dir):
        raise Exception("input_dir {0} not found".format(input_dir))

    dirpath, dirnames, filenames = next(os.walk(input_dir))
    if suffix is None:
        for f in filenames:
            for s in [".png", ".jpg", ".jpeg"]:
                if f.endswith(s):
                    suffix = s
                    break
            else:
                raise Exception("cannot infer suffix, please specify")
            if suffix is not None:
                break
    filenames = sorted(list(filter(lambda x: x.endswith(suffix), filenames)))
    if img_type is None:
        img_dtype = tables.UInt8Atom()
    else:
        raise Exception("Unknown img_type, only support uint8")
    if img_shape is None:
        img_shape = np.array(Image.open(filenames[0])).shape
    if not output_name.endswith(".h5"):
        output_name = output_name + '.h5'

    hdf5_file = tables.open_file(output_name, mode='w')
    storage = hdf5_file.create_earray(hdf5_file.root, 'images', img_dtype, shape=data_shape)

    for f in filenames:
        img = np.array(Image.open(os.path.join(input_dir, f)), dtype=np.uint8)
        storage.append(img[None])


DATA_DIR = "/data/ziz/not-backed-up/jxu/CelebA/celebA"

convert_images_dir_to_h5(DATA_DIR, "celeba")
