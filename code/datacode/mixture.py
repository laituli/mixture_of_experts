import cv2
import numpy as np
from datacode import cifar


def transform_mnist_x(X):
    X = np.reshape(X,(-1,28,28,1))
    X = np.tile(X,(1,1,1,3))
    X = np.array([cv2.resize(x, dsize=(32, 32), interpolation=cv2.INTER_CUBIC) for x in X])
    return X

def mnist_onehot(Y):
    return np.equal(Y[:,np.newaxis],np.arange(10)[np.newaxis,:]) * 1.

def cifar100_y_to_mixture_y(y):
    y = np.pad(y,[(0,0),(10,0)],mode="constant")
    return y


class DataSet:
    def __init__(self,x,y,class_names):
        self.x = x
        self.y = y
        self.class_count = y.shape[1]
        self.class_names = class_names
        self.N = len(x)

    def get_names(self, y):
        return self.class_names[y]



class Mixture(DataSet):

    def mix_class_to_set_class(self,mc):
        tmp = np.int(mc[:,np.newaxis] < self.cum_classes[np.newaxis,:])
        s = np.sum(tmp,axis=1)-1
        c = mc-self.cum_classes[s]
        return s,c

    def set_class_to_mix_class(self,s,c):
        return self.cum_classes[s] + c

    def __init__(self,datasets):
        class_counts = [dataset.class_count for dataset in datasets]
        self.cum_classes = np.concatenate([[0],np.cumsum(class_counts)])
        self.class_count = self.cum_classes[-1]

        x = np.concatenate([dataset.x for dataset in datasets])
        ys = []
        for i,dataset in enumerate(datasets):
            left = self.cum_classes[i]
            right = self.class_count - left - dataset.class_count
            y = np.pad(dataset.y, ((0,0),(left,right)),'constant')
            ys.append(y)
        y = np.concatenate(ys)
        class_names = np.concatenate([dataset.class_names for dataset in datasets])

        DataSet.__init__(self,x,y,class_names)





mnist_train_x, mnist_test_x = map(transform_mnist_x, (mnist.train.images,mnist.test.images))
mnist_train_y, mnist_test_y = map(mnist_onehot,(mnist.train.labels,mnist.test.labels))

mnist_train = DataSet(mnist_train_x, mnist_train_y, [str(i) for i in range(10)])
mnist_test = DataSet(mnist_test_x, mnist_test_y, [str(i) for i in range(10)])
cifar100_train = DataSet(cifar.images, cifar.fine_labels, cifar.label_list)
cifar100_test = DataSet(cifar.test_images, cifar.test_fine, cifar.label_list)
mixture_train = Mixture([mnist_train,cifar100_train])
mixture_test = Mixture([mnist_test,cifar100_test])

print("mixture shapes:")
print(np.shape(mixture_train.x))
print(np.shape(mixture_train.y))
print(np.shape(mixture_test.x))
print(np.shape(mixture_test.y))
print()
