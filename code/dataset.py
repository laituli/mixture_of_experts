import cv2
import numpy as np
import matplotlib.pyplot as plt

def onehot(labels):
    l = np.array(labels)
    n = np.max(l)
    return np.equal(l[:,np.newaxis], np.arange(n+1)[np.newaxis,:]) * np.float32(1)

def check_shape(X):
    # 4 cases of image batch shape
    # BHWC
    # BHW
    # BSC
    # BS
    if len(X.shape) == 4:
        H, W, C = X.shape[1:]
    elif len(X.shape) == 2:
        S, C = X.shape[1], 1
        H = W = int(np.sqrt(S))
    elif len(X.shape) == 3:
        if X.shape[2] in [1, 3]:
            S, C = X.shape[1:]
            H = W = int(np.sqrt(S))
        else:
            C = 1
            H, W = X.shape[1:]
    else:
        raise Exception('cannot understand image shape')
    return H,W,C

def resized_images(X,new_H,new_W, is_gray):
    H,W,C = check_shape(X)
    X = np.reshape(X, (-1,H,W,C))
    if is_gray and C == 3:
        X = np.mean(X,axis=-1,keepdims=True)
    elif not is_gray and C == 1:
        X = np.tile(X, (1, 1, 1, 3))
    X = np.array([cv2.resize(x, dsize=(new_H, new_W), interpolation=cv2.INTER_CUBIC) for x in X])
    if is_gray: X = X[:,:,:,np.newaxis]
    return X

class Dataset:

    def __init__(self,train_X,train_Y,test_X,test_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

    def print_shapes(self):
        print('train X',self.train_X.shape)
        print('train Y',self.train_Y.shape)
        print('test X',self.test_X.shape)
        print('test Y',self.test_Y.shape)

def combine_datasets(sets,H,W, is_gray):
    train_X = np.concatenate([resized_images(set.train_X,H,W,is_gray) for set in sets])
    test_X = np.concatenate([resized_images(set.test_X,H,W,is_gray) for set in sets])

    train_Y = np.zeros((0,0))
    test_Y = np.zeros((0,0))
    for set in sets:
        a = train_Y.shape[1]
        b = set.train_Y.shape[1]

        train_Y_a = np.pad(train_Y,[(0,0),(0,b)],mode='constant')
        train_Y_b = np.pad(set.train_Y,[(0,0),(a,0)],mode='constant')
        train_Y = np.concatenate([train_Y_a,train_Y_b])

        test_Y_a = np.pad(test_Y, [(0, 0), (0, b)], mode='constant')
        test_Y_b = np.pad(set.test_Y, [(0, 0), (a, 0)], mode='constant')
        test_Y = np.concatenate([test_Y_a, test_Y_b])

    return Dataset(train_X,train_Y,test_X,test_Y)


def visualize(image):
    # BHWC

    cols = int(np.ceil(np.sqrt(image.shape[0])))
    if image.shape[-1] == 1:
        image = np.tile(image,(1,1,1,3))

    img_number = 0
    for row in range(0, cols):
        for col in range(0, cols):
            if img_number > image.shape[0] - 1: break
            ax = plt.subplot2grid((cols, cols), (row, col))
            ax.axes.axes.get_xaxis().set_visible(False)
            ax.axes.axes.get_yaxis().set_visible(False)

            imgplot = ax.imshow(image[img_number])
            imgplot.set_interpolation('nearest')
            ax.xaxis.set_ticks_position('top')
            ax.yaxis.set_ticks_position('left')
            img_number += 1
    plt.show()