
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
from datacode.dataset import Dataset, onehot

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
    if classes not in [10,20,100, 120]:
        raise Exception("cifar can have only 10, 20 or 100 classes\nor 100 with 20 super classes by input 120")

    if classes == 10:
        path = "data/cifar-10/"
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        folder = "cifar-10-batches-py"
    else:
        path = "data/cifar-100/"
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

def cifar_meta(classes):
    if classes not in [10,100]:
        raise Exception("cifar can have only 10 or 100 classes")

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

    z = train_data[b'coarse_labels']
    y = train_data[b'fine_labels']

    print(z[:10])
    print(y[:10])

    """
    meta = unpickle(path,folder,"meta")
    label_list = meta[b'fine_label_names']
    print("fine label list", label_list)
    label_list = meta[b'coarse_label_names']
    print("corase label names", label_list)
    """
    #get label names as below: (cifar100)
    #meta = unpickle("meta")
    #label_list = meta[b'fine_label_names']



def cifar100_labels():

    path = "../data/cifar-100/"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    folder = "cifar-100-python"

    print("maybe download cifar")
    maybe_download_and_extract(url, path)
    print("preprocessing cifar data")


    #train_data, test_data = unpickle(path,folder,"train"), unpickle(path,folder,"test")
    #z = train_data[b'coarse_labels']
    #y = train_data[b'fine_labels']

    #print(z[:10])
    #print(y[:10])

    meta = unpickle(path,folder,"meta")

    label_list = meta[b'fine_label_names']
    label_list = [label.decode("UTF-8") for label in label_list]
    print("fine label list")
    for ilabel in enumerate(label_list):
        print("%i\t%s" % ilabel)
    print()


    super_label_list = meta[b'coarse_label_names']
    super_label_list = [label.decode("UTF-8") for label in super_label_list]
    print("corase label names")
    for ilabel in enumerate(super_label_list):
        print("%i\t%s" % ilabel)
    print()

    cifar100 = cifar(100)
    cifar20 = cifar(20)


    sets = [set() for label in super_label_list]
    for i in range(len(cifar100.test_Y)):
        y = cifar100.test_Y[i]
        z = cifar20.test_Y[i]
        y = np.argmax(y)
        z = np.argmax(z)
        super_set = sets[int(z)]
        super_set.add(y)
    for i,s in enumerate(sets):
        super_label = super_label_list[i]
        s = tuple(sorted(s))
        s = (i, super_label) + s
        print("%i\t%s\t%i\t%i\t%i\t%i\t%i" % s)
    print("finished")


#cifar100_labels()


def cifar100_grouped():
    path = "../data/cifar-100/"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    folder = "cifar-100-python"

    print("maybe download cifar")
    maybe_download_and_extract(url, path)
    print("preprocessing cifar data")

    train_data, test_data = unpickle(path, folder, "train"), unpickle(path, folder, "test")
    #meta = unpickle(path,folder,"meta")

    train_X, test_X = train_data[b'data'], test_data[b'data']
    train_Y_old, test_Y_old = train_data[b'fine_labels'], test_data[b'fine_labels']
    train_Z, test_Z = train_data[b'coarse_labels'], test_data[b'coarse_labels']

    print("grouping y to z")

    coarse2fine = [set() for _ in range(20)]
    for i in range(len(test_X)):
        y,z = test_Y_old[i], test_Z[i]
        coarse2fine[z].add(y)
    new2old_fine = [l for s in coarse2fine for l in sorted(s)]
    old2new_fine = [new2old_fine.index(i) for i in range(len(new2old_fine))]

    train_Y_new = [old2new_fine[old] for old in train_Y_old]
    test_Y_new = [old2new_fine[old] for old in test_Y_old]

    print("converting x")
    train_X, test_X = map(convert_cifar_images,(train_X, test_X))
    train_Y, test_Y = map(onehot,(train_Y_new,test_Y_new))
    train_Z, test_Z = map(onehot,(train_Z,test_Z))

    dataset = Dataset(train_X, train_Y, test_X, test_Y)
    dataset.train_Z = train_Z
    dataset.test_Z = test_Z
    print("cifar100 ready")
    dataset.print_shapes()
    return dataset


#for y,z in zip(cifar100.test_Y, cifar100.test_Z):
#    print(np.argmax(y),np.argmax(z))
