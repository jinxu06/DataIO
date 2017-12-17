# http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html
import numpy as np
import tables
import os

data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow
img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved

# check the order of data and chose proper data shape to save images
if data_order == 'th':
    data_shape = (0, 3, 224, 224)
elif data_order == 'tf':
    data_shape = (0, 224, 224, 3)

# open a hdf5 file and create earrays

hdf5_path = "celeba.h5"

hdf5_file = tables.open_file(hdf5_path, mode='w')

train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
val_storage = hdf5_file.create_earray(hdf5_file.root, 'val_img', img_dtype, shape=data_shape)
test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_img', img_dtype, shape=data_shape)

# mean_storage = hdf5_file.create_earray(hdf5_file.root, 'train_mean', img_dtype, shape=data_shape)

# create the label arrays and copy the labels data in them
# hdf5_file.create_array(hdf5_file.root, 'train_labels', train_labels)
# hdf5_file.create_array(hdf5_file.root, 'val_labels', val_labels)
# hdf5_file.create_array(hdf5_file.root, 'test_labels', test_labels)

DATA_DIR = "/data/ziz/not-backed-up/jxu/CelebA/celebA"

dirpath, dirnames, filenames = next(os.walk(DATA_DIR))
filenames = sorted(list(filter(lambda x: x.endswith(".jpg"), filenames)))
print(filenames)
