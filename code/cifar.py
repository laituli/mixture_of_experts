
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import os
import urllib
import urllib.request
import zipfile
import tarfile
from dataset import Dataset, onehot

img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels

def convert_cifar_images(raw):
    raw_float = np.array(raw, dtype=np.float32) / 255.0
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])
    return images

def maybe_download_and_extract(download_from, download_to):
    filename = download_from.split('/')[-1]
    file_path = os.path.join(download_to, filename)

    if not os.path.exists(file_path):
        if not os.path.exists(download_to):
            os.makedirs(download_to)
        file_path, _ = urllib.request.urlretrieve(url=download_from,
                                                  filename=file_path)
        print("downloaded data")

        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_to)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(download_to)
        print("extracted data")
    else:
        print("data already exist")

def unpickle(path,folder,file):
    file_path = os.path.join(path, folder, file)
    print("processing",file_path)
    with open(file_path, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def cifar(classes):
    if classes not in [10,20,100]:
        raise Exception("cifar can have only 10, 20 or 100 classes")

    if classes == 10:
        path = "../data/cifar-10/"
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        folder = "cifar-10-batches-py"
    else:
        path = "../data/cifar-100/"
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        folder = "cifar-100-python"

    print("maybe download cifar")
    maybe_download_and_extract(url, path)
    print("preprocessing cifar data")

    if classes == 10:
        train_data = {b'data':np.zeros((0,3072)),b'labels':[]}
        for b in range(1,6):
            batch = unpickle(path,folder,"data_batch_"+str(b))
            train_data[b'data'] = np.concatenate((train_data[b'data'],batch[b'data']))
            train_data[b'labels'] += batch[b'labels']
            #print(np.shape(train_data[b'data']),np.shape(train_data[b'labels']))
        test_data = unpickle(path,folder,"test_batch")
    else:
        train_data, test_data = unpickle(path,folder,"train"), unpickle(path,folder,"test")


    if classes == 10:
        train_X, train_Y = train_data[b'data'], train_data[b'labels']
        test_X, test_Y = test_data[b'data'], test_data[b'labels']
    elif classes == 20:
        train_X, train_Y = train_data[b'data'], train_data[b'coarse_labels']
        test_X, test_Y = test_data[b'data'], test_data[b'coarse_labels']
    elif classes == 100:
        train_X, train_Y = train_data[b'data'], train_data[b'fine_labels']
        test_X, test_Y = test_data[b'data'], test_data[b'fine_labels']
    else:
        raise Exception("cifar can have only 10, 20 or 100 classes")

    train_X, test_X = map(convert_cifar_images,(train_X, test_X))
    train_Y, test_Y = map(onehot,(train_Y,test_Y))

    cifar = Dataset(train_X,train_Y,test_X,test_Y)


    print("cifar-"+str(classes)+"data ready")
    cifar.print_shapes()

    return cifar

    #get label names as below: (cifar100)
    #meta = unpickle("meta")
    #label_list = meta[b'fine_label_names']

