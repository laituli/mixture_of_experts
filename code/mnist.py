import numpy as np
from dataset import Dataset, onehot

USE_TENSORFLOW_MNIST = False

if USE_TENSORFLOW_MNIST:


    from tensorflow.examples.tutorials.mnist import input_data

    data_directory = "../data"
    mnist = input_data.read_data_sets(data_directory)

    mnist_train_y, mnist_test_y = map(onehot,(mnist.train.labels, mnist.test.labels))
    mnist = Dataset(mnist.train.images, mnist_train_y, mnist.test.images, mnist_test_y)
    #print("mnist:")
    #mnist.print_shapes()

else:

    import urllib.request
    import gzip
    import pickle
    import os


    def _download(file_name):
        file_path = dataset_dir + "/" + file_name

        if os.path.exists(file_path):
            return

        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(url_base + file_name, file_path)
        print("Done")

    def download_mnist():
        for v in key_file.values():
           _download(v)

    def _load_label(file_name):
        file_path = dataset_dir + "/" + file_name

        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_path, 'rb') as f:
                labels = np.frombuffer(f.read(), np.uint8, offset=8)
        print("Done")

        return labels

    def _load_img(file_name):
        file_path = dataset_dir + "/" + file_name

        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_path, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, img_size)
        print("Done")

        return data

    def _convert_numpy():
        dataset = {}
        dataset['train_img'] =  _load_img(key_file['train_img'])
        dataset['train_label'] = _load_label(key_file['train_label'])
        dataset['test_img'] = _load_img(key_file['test_img'])
        dataset['test_label'] = _load_label(key_file['test_label'])

        return dataset

    def init_mnist():
        download_mnist()
        dataset = _convert_numpy()
        print("Creating pickle file ...")
        with open(save_file, 'wb') as f:
            pickle.dump(dataset, f, -1)
        print("Done")

    def _change_one_hot_label(X):
        T = np.zeros((X.size, 10))
        for idx, row in enumerate(T):
            row[X[idx]] = 1

        return T

    def load_mnist(normalize=True, flatten=False, one_hot_label=True):
        if not os.path.exists(save_file):
            init_mnist()

        with open(save_file, 'rb') as f:
            dataset = pickle.load(f)

        if normalize:
            for key in ('train_img', 'test_img'):
                dataset[key] = dataset[key].astype(np.float32)
                dataset[key] /= 255.0

        if not flatten:
             for key in ('train_img', 'test_img'):
                dataset[key] = dataset[key].reshape(-1, 28, 28, 1)

        if one_hot_label:
            dataset['train_label'] = onehot(dataset['train_label'])
            dataset['test_label'] = onehot(dataset['test_label'])

        return Dataset(dataset['train_img'], dataset['train_label'], dataset['test_img'], dataset['test_label'])




    # Load the MNIST dataset
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    key_file = {
        'train_img':'train-images-idx3-ubyte.gz',
        'train_label':'train-labels-idx1-ubyte.gz',
        'test_img':'t10k-images-idx3-ubyte.gz',
        'test_label':'t10k-labels-idx1-ubyte.gz'
    }

    dataset_dir = "../data/mnist/"
    save_file = dataset_dir + "mnist.pkl"

    train_num = 60000
    test_num = 10000
    #img_dim = (28, 28, 1)
    img_size = 784

    print("load mnist")
    mnist = load_mnist()


mnist.print_shapes()
print(np.sum(mnist.train_Y,axis=0))
print(np.sum(mnist.test_Y,axis=0))
