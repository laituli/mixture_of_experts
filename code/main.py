#import itertools as iter
import functools as func

from mnist import mnist
from cifar import cifar
from dataset import combine_datasets, visualize
import time
import numpy as np
np.random.seed(12345)
import tensorflow as tf
tf.random.set_random_seed(12345)



# DEFINE DATA
image_height = image_width = 32
original_datasets = [mnist, cifar(10)] # cifar10, 20 or 100 is allowed
#original_datasets = [cifar(10)]
dataset = combine_datasets(original_datasets, image_width, image_height)

#visualize(dataset.train_X[:25])
visualize(dataset.train_X[54990:55015])
visualize(dataset.train_X[np.argmax(dataset.train_Y,axis=1)==3][:25])
visualize(dataset.test_X[np.argmax(dataset.test_Y,axis=1)==7][:25])

print("using dataset:")
dataset.print_shapes()

train_size = dataset.train_X.shape[0]
test_size = dataset.test_X.shape[0]
num_classes = dataset.train_Y.shape[1]
print("number of classes:",num_classes)

# DEFINE TOPOLOGY
GPU = '/job:localhost/replica:0/task:0/device:GPU:0'
with tf.device(GPU):
    X = tf.placeholder(tf.float32, [None, image_height, image_width, 3], 'X')
    Y_Onehot = tf.placeholder(tf.float32, [None, num_classes], 'Y_Onehot')
    Training = tf.placeholder_with_default(False, ())

    net_name = "mix net sum attention-gate"

    # normal net will configured to be the baseline
    # some version of mix net with softmax will become another baseline
    # nets below are just to have some minor test

    if net_name == "normal net":
        layer = X
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.max_pooling2d(layer,(2,2),(2,2),padding='same')
        layer = tf.layers.conv2d(layer, 128, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer, 128, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.max_pooling2d(layer,(2,2),(2,2),padding='same')
        layer = tf.layers.conv2d(layer, 256, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer, 256, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.max_pooling2d(layer,(2,2),(2,2),padding='same')
        layer = tf.layers.flatten(layer)
        layer = tf.layers.dense(layer,256,tf.nn.relu)
        layer = tf.layers.dropout(layer,training=Training)
        layer = tf.layers.dense(layer,num_classes)
        Y_Logits = layer
    # tested running time with submodules without gating with reduce-sum / sum / concat
    # sum is a bit faster
    elif net_name == "test1: no-gate net reduce-sum":
        layer = X
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        splits = 2
        layers = []
        for m in range(splits):
            sub_layer = layer
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            layers.append(sub_layer)
        layer = tf.reduce_sum(layers, axis=0)
        # layer = sum(layers)
        # layer = tf.concat(layer, axis=-1)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        splits = 4
        gate = tf.reduce_mean(layer, [1, 2])
        gate = tf.layers.dense(gate, splits, tf.nn.softmax)
        layers = []
        for m in range(splits):
            sub_layer = layer
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            layers.append(sub_layer)
        layer = tf.reduce_sum(layers, axis=0)
        # layer = sum(layers)
        # layer = tf.concat(layer,axis=-1)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        layer = tf.layers.flatten(layer)
        layer = tf.layers.dense(layer, 256, tf.nn.relu)
        layer = tf.layers.dropout(layer, training=Training)
        layer = tf.layers.dense(layer, num_classes)
        Y_Logits = layer
    elif net_name == "test1: no-gate net sum":  # test running time
        layer = X
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        splits = 2
        layers = []
        for m in range(splits):
            sub_layer = layer
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            layers.append(sub_layer)
        layer = sum(layers)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        splits = 4
        gate = tf.reduce_mean(layer, [1, 2])
        gate = tf.layers.dense(gate, splits, tf.nn.softmax)
        layers = []
        for m in range(splits):
            sub_layer = layer
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            layers.append(sub_layer)
        layer = sum(layers)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        layer = tf.layers.flatten(layer)
        layer = tf.layers.dense(layer, 256, tf.nn.relu)
        layer = tf.layers.dropout(layer, training=Training)
        layer = tf.layers.dense(layer, num_classes)
        Y_Logits = layer
    elif net_name == "test1: no-gate net concat":  # test running time
        layer = X
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        splits = 2
        layers = []
        for m in range(splits):
            sub_layer = layer
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            layers.append(sub_layer)
        layer = tf.concat(layers, axis=-1)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        splits = 4
        gate = tf.reduce_mean(layer, [1, 2])
        gate = tf.layers.dense(gate, splits, tf.nn.softmax)
        layers = []
        for m in range(splits):
            sub_layer = layer
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            layers.append(sub_layer)
        layer = tf.concat(layers, axis=-1)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        layer = tf.layers.flatten(layer)
        layer = tf.layers.dense(layer, 256, tf.nn.relu)
        layer = tf.layers.dropout(layer, training=Training)
        layer = tf.layers.dense(layer, num_classes)
        Y_Logits = layer
    # tested  concat-and-einsum/multiply-and-sum running time
    # concat-and-einsum is much slower
    elif net_name == "test2: mix net einsum":
        layer = X
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        splits = 2
        gate = tf.reduce_mean(layer, [1, 2])
        gate = tf.layers.dense(gate, splits, tf.nn.softmax)
        layers = []
        for m in range(splits):
            sub_layer = layer
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer = sub_layer[tf.newaxis]
            layers.append(sub_layer)
        layer = tf.concat(layers, axis=0)
        layer = tf.einsum('bm,mbhwc->bhwc', gate, layer)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        splits = 4
        gate = tf.reduce_mean(layer, [1, 2])
        gate = tf.layers.dense(gate, splits, tf.nn.softmax)
        layers = []
        for m in range(splits):
            sub_layer = layer
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer = sub_layer[tf.newaxis]
            layers.append(sub_layer)
        layer = tf.concat(layers, axis=0)
        layer = tf.einsum('bm,mbhwc->bhwc', gate, layer)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        layer = tf.layers.flatten(layer)
        layer = tf.layers.dense(layer, 256, tf.nn.relu)
        layer = tf.layers.dropout(layer, training=Training)
        layer = tf.layers.dense(layer, num_classes)
        Y_Logits = layer
    elif net_name == "test2: mix net sum":
        layer = X
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        splits = 2
        gate = tf.reduce_mean(layer, [1, 2])
        gate = tf.layers.dense(gate, splits, tf.nn.softmax)
        layers = []
        for m in range(splits):
            sub_layer = layer
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer *= gate[:, m, tf.newaxis, tf.newaxis, tf.newaxis]
            layers.append(sub_layer)
        layer = sum(layers)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        splits = 4
        gate = tf.reduce_mean(layer, [1, 2])
        gate = tf.layers.dense(gate, splits, tf.nn.softmax)
        layers = []
        for m in range(splits):
            sub_layer = layer
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer *= gate[:, m, tf.newaxis, tf.newaxis, tf.newaxis]
            layers.append(sub_layer)
        layer = sum(layers)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        layer = tf.layers.flatten(layer)
        layer = tf.layers.dense(layer, 256, tf.nn.relu)
        layer = tf.layers.dropout(layer, training=Training)
        layer = tf.layers.dense(layer, num_classes)
        Y_Logits = layer
    elif net_name == "mix net sum attention-gate":
        layer = X
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        splits = 2
        gate = tf.einsum('bhwc,c->bhw', layer, tf.Variable(tf.zeros(64)))
        # gate = tf.layers.dense(layer, 1, use_bias=False)  # BHWC -> BHW1
        # gate = tf.nn.softmax(gate,axis=[1,2])
        gate -= tf.reduce_max(gate, axis=[1, 2], keepdims=True)
        gate = tf.exp(gate)
        gate /= tf.reduce_sum(gate, axis=[1, 2], keepdims=True)
        # softmax(A_i) == exp(A_i) / sum_j (exp(A_j)) == exp(A_i-max(A)) / sum_j (exp(A_j-max(A)))
        gate = gate[:,:,:,tf.newaxis]
        gate *= layer
        gate = tf.reduce_sum(layer, [1, 2])  # BHWC -> BC
        gate = tf.layers.dense(gate, splits, tf.nn.softmax)
        layers = []
        for m in range(splits):
            sub_layer = layer
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer *= gate[:,m,tf.newaxis,tf.newaxis,tf.newaxis]
            layers.append(sub_layer)
        layer = sum(layers)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        splits = 4
        gate = tf.einsum('bhwc,c->bhw', layer, tf.Variable(tf.zeros(64)))
        # gate = tf.layers.dense(layer, 1, use_bias=False)  # BHWC -> BHW1
        # gate = tf.nn.softmax(gate,axis=[1,2])
        gate -= tf.reduce_max(gate, axis=[1, 2], keepdims=True)
        gate = tf.exp(gate)
        gate /= tf.reduce_sum(gate, axis=[1, 2], keepdims=True)
        # softmax(A_i) == exp(A_i) / sum_j (exp(A_j)) == exp(A_i-max(A)) / sum_j (exp(A_j-max(A)))
        gate = gate[:, :, :, tf.newaxis]
        gate *= layer
        gate = tf.reduce_sum(layer, [1, 2])  # BHWC -> BC
        gate = tf.layers.dense(gate, splits, tf.nn.softmax)
        layers = []
        for m in range(splits):
            sub_layer = layer
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer *= gate[:, m, tf.newaxis, tf.newaxis, tf.newaxis]
            layers.append(sub_layer)
        layer = sum(layers)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        layer = tf.layers.flatten(layer)
        layer = tf.layers.dense(layer, 256, tf.nn.relu)
        layer = tf.layers.dropout(layer, training=Training)
        layer = tf.layers.dense(layer, num_classes)
        Y_Logits = layer
    elif net_name == "hierarchical":
        layer = X
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        splits = 2
        gate = tf.reduce_mean(layer, [1, 2])
        gate = tf.layers.dense(gate, splits, tf.nn.softmax)
        layers = []
        for m in range(splits):
            sub_layer = layer
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)
            sub_layer = tf.layers.conv2d(sub_layer, 64, (3, 3), padding='same', activation=tf.nn.relu)

            sub_layer = tf.layers.max_pooling2d(sub_layer, (2, 2), (2, 2), padding='same')
            sub_splits = 2
            sub_gate = tf.reduce_mean(sub_layer, [1,2])
            sub_gate = tf.layers.dense(sub_gate, sub_splits, tf.nn.softmax)
            subsub_layers = []
            for sub_m in range(sub_splits):
                subsub_layer = sub_layer
                subsub_layer = tf.layers.conv2d(subsub_layer, 64, (3,3), padding='same', activation=tf.nn.relu)
                subsub_layer = tf.layers.conv2d(subsub_layer, 64, (3,3), padding='same', activation=tf.nn.relu)
                subsub_layer *= sub_gate[:,sub_m,tf.newaxis,tf.newaxis,tf.newaxis]
                subsub_layers.append(subsub_layer)
            sub_layer = sum(subsub_layers)
            sub_layer *= gate[:, m, tf.newaxis, tf.newaxis, tf.newaxis]
            layers.append(sub_layer)
        layer = sum(layers)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')

        layer = tf.layers.flatten(layer)
        layer = tf.layers.dense(layer, 256, tf.nn.relu)
        layer = tf.layers.dropout(layer, training=Training)
        layer = tf.layers.dense(layer, num_classes)
        Y_Logits = layer
    # used to debug
    elif net_name == "random":
        Y_Logits = tf.random.normal(tf.shape(Y_Onehot)) + tf.Variable(0.) # +var for backprop to run properly
    else:
        raise Exception('unimplemented net:'+net_name)

    # DEFINE MEASURES
    Loss = tf.losses.softmax_cross_entropy(Y_Onehot,Y_Logits)

    Y_True = tf.argmax(Y_Onehot,axis=1)
    Y_Pred = tf.argmax(Y_Logits,axis=1)
    Acc, Acc_op = tf.metrics.accuracy(Y_True,Y_Pred)
    print(Acc,Acc_op)
    # and TIME: converging time, prediction time


# DEFINE TRAINING
epochs = 20
batchsize = 128

Trainstep = tf.train.AdamOptimizer().minimize(Loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    summary = tf.Summary()
    print(summary)

    train_acc_history = []
    start = time.time()
    for e in range(epochs):
        print("epoch",e)
        sess.run(tf.local_variables_initializer()) # reset acc
        indices = np.arange(train_size)
        np.random.shuffle(indices)
        for b in range(0,train_size,batchsize):
            if b%12800 == 0:
                print("batch",b)
            batch_i = indices[b:b+batchsize]
            batch_x = dataset.train_X[batch_i]
            batch_y = dataset.train_Y[batch_i]
            feeddict = {X: batch_x, Y_Onehot: batch_y, Training:True}
            sess.run((Trainstep,Acc_op),feeddict)
        print()
        train_acc_history.append(sess.run(Acc))
        print('epoch',e,'train acc',train_acc_history[-1])

    end = time.time()
    train_time = end-start

    # DEFINE TESTING

    start = time.time()
    sess.run(tf.local_variables_initializer()) # reset acc
    indices = np.arange(test_size)
    for b in range(0, test_size, batchsize):
        batch_i = indices[b:b + batchsize]
        batch_x = dataset.test_X[batch_i]
        batch_y = dataset.test_Y[batch_i]
        feeddict = {X: batch_x, Y_Onehot: batch_y}
        sess.run(Acc_op, feeddict)
    end = time.time()
    test_time = end-start
    test_acc = sess.run(Acc)

lines = [
    '\n',net_name,
    '\nepochs ',str(epochs),
    '\ntrain time ',str(train_time),
    '\ntrain acc by epoch ',str(train_acc_history),
    '\ntest time ',str(test_time),
    '\ntest acc ',str(test_acc),
    '\n'
]

with open("results.txt",'a+') as file:
    file.writelines(lines)



