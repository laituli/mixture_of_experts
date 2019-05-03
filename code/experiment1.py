#import itertools as iter
import functools as func

from mnist import mnist
from cifar import cifar
from omniglot import omniglot
from dataset import combine_datasets, visualize
import time
import numpy as np
np.random.seed(12345)
import tensorflow as tf
tf.random.set_random_seed(12345)
import matplotlib.pyplot as plt


# DEFINE DATA
image_height = image_width = 32
is_gray = True
#original_datasets = [cifar(100)]
#original_datasets = [omniglot]
original_datasets = [mnist, cifar(10)] # cifar10, 20 or 100 is allowed

#original_datasets = [mnist]
dataset = combine_datasets(original_datasets, image_width, image_height, is_gray)


#visualize(dataset.train_X[:100], width=image_width)

def visualize_class(c,n=36):
    visualize(dataset.train_X[np.argmax(dataset.train_Y, axis=1) == c][:n])
    visualize(dataset.test_X[np.argmax(dataset.test_Y, axis=1) == c][:n])


#visualize_class(9)
#visualize_class(10)

print("using dataset:")
dataset.print_shapes()

train_size = dataset.train_X.shape[0]
test_size = dataset.test_X.shape[0]
num_classes = dataset.train_Y.shape[1]
print("number of classes:",num_classes)


GPU = '/job:localhost/replica:0/task:0/device:GPU:0'
with tf.device(GPU):
    X = tf.placeholder(tf.float32, [None, image_height, image_width, 1 if is_gray else 3], 'X')
    Y_Onehot = tf.placeholder(tf.float32, [None, num_classes], 'Y_Onehot')
    Training = tf.placeholder_with_default(False, ())

    Y_True = tf.argmax(Y_Onehot, axis=1)
    Is_Mnist = Y_True < 10

    #nets = ['baseline','cnngate','fcgate']
    nets = ['cnngate','baseline','fcgate']
    results = dict()

    def sub_NN(X,Training):
        layer = X
        layer = tf.layers.conv2d(layer, 128, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')
        layer = tf.layers.conv2d(layer, 128, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')
        layer = tf.layers.flatten(layer)
        layer = tf.layers.dense(layer, 128, tf.nn.relu)
        #layer = tf.layers.dropout(layer, training=Training)
        layer = tf.layers.dense(layer, num_classes)
        Y_Logits = layer
        return Y_Logits

    for net in nets:
        NN1_out = sub_NN(X,Training)
        NN2_out = sub_NN(X,Training)
        if net == 'baseline':
            gate = tf.cast(Is_Mnist,tf.float32)[:,tf.newaxis] * np.array([[1,-1]]) + np.array([[0,1]])
        elif net == 'cnngate':
            layer = X
            layer = tf.layers.conv2d(layer,64,(3,3),padding='same')
            layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')
            layer = tf.layers.conv2d(layer,64,(3,3),padding='same')
            layer = tf.layers.max_pooling2d(layer, (2, 2), (2, 2), padding='same')
            layer = tf.layers.flatten(layer)
            layer = tf.layers.dense(layer, 64, tf.nn.relu)
            layer = tf.layers.dense(layer, 2, tf.nn.softmax)
            gate = layer
        elif net == 'fcgate':
            layer = X
            layer = tf.layers.flatten(layer)
            layer = tf.layers.dense(layer, 64, tf.nn.relu)
            layer = tf.layers.dense(layer, 2, tf.nn.softmax)
            gate = layer
        else:
            raise Exception('unimplemented net:'+net)
        Y_Logits = (gate[:,0:1] * NN1_out) + (gate[:,1:2] * NN2_out)
        Loss = tf.losses.softmax_cross_entropy(Y_Onehot,Y_Logits)

        Y_Pred = tf.argmax(Y_Logits,axis=1)
        Acc, Acc_op = tf.metrics.accuracy(Y_True,Y_Pred)
        print(Acc,Acc_op)

        Mnist_gate = tf.boolean_mask(gate,Is_Mnist)
        Cifar_gate = tf.boolean_mask(gate,~Is_Mnist)
        gate_mean = {
           'mnist': {0: tf.metrics.mean(Mnist_gate[:,0]),
                     1: tf.metrics.mean(Mnist_gate[:,1])},
           'cifar': {0: tf.metrics.mean(Cifar_gate[:,0]),
                     1: tf.metrics.mean(Cifar_gate[:,1])}
        }
        gate_mean_val = {
            'mnist': {0: gate_mean['mnist'][0][0],
                      1: gate_mean['mnist'][1][0]},
            'cifar': {0: gate_mean['cifar'][0][0],
                      1: gate_mean['cifar'][1][0]},
        }
        gate_mean_op = {
            'mnist': {0: gate_mean['mnist'][0][1],
                      1: gate_mean['mnist'][1][1]},
            'cifar': {0: gate_mean['cifar'][0][1],
                      1: gate_mean['cifar'][1][1]},
        }
        del gate_mean

        # and TIME: converging time, prediction time

        epochs = 1000
        batchsize = 128

        Trainstep = tf.train.AdamOptimizer().minimize(Loss)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            summary = tf.Summary()
            print(summary)

            train_acc_history = []
            test_acc_history = []
            test_times = []

            start = time.time()
            for e in range(epochs):
                print()
                print("epoch",e)
                sess.run(tf.local_variables_initializer()) # reset acc
                indices = np.arange(train_size)
                #np.random.shuffle(indices)
                for b in range(0,train_size,batchsize):
                    if b%12800 == 0:
                        print("batch",b)
                    batch_i = indices[b:b+batchsize]
                    batch_x = dataset.train_X[batch_i]
                    batch_y = dataset.train_Y[batch_i]
                    feeddict = {X: batch_x, Y_Onehot: batch_y, Training:True}
                    sess.run((Trainstep,Acc_op),feeddict)
                train_acc_history.append(sess.run(Acc))
                print('epoch',e,'train acc',train_acc_history[-1])

                test_start = time.time()
                sess.run(tf.local_variables_initializer())  # reset acc
                indices = np.arange(test_size)
                for b in range(0, test_size, batchsize):
                    batch_i = indices[b:b + batchsize]
                    batch_x = dataset.test_X[batch_i]
                    batch_y = dataset.test_Y[batch_i]
                    feeddict = {X: batch_x, Y_Onehot: batch_y}
                    sess.run(Acc_op, feeddict)
                test_end = time.time()
                test_acc = sess.run(Acc)
                test_time = test_end - test_start
                test_acc_history.append(test_acc)
                test_times.append(test_time)
                print('epoch',e,'test acc',test_acc_history[-1])

                earlystop = e - np.argmax(test_acc_history) > 5
                if earlystop:
                    epochs = e + 1
                    break

            end = time.time()
            train_time = end-start-sum(test_times)

            # plot
            plt.plot(range(len(train_acc_history)),train_acc_history,'b')
            plt.plot(range(len(test_acc_history)),test_acc_history,'r')
            #plt.axis([0,epochs,0,1])
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.show()

            """compute gate table"""
            gate_mean_table = dict()
            if True:
                sess.run(tf.local_variables_initializer())  # reset gate_mean
                indices = np.arange(train_size)
                for b in range(0, train_size, batchsize):
                    batch_i = indices[b:b + batchsize]
                    batch_x = dataset.train_X[batch_i]
                    batch_y = dataset.train_Y[batch_i]
                    feeddict = {X: batch_x, Y_Onehot: batch_y}
                    sess.run(gate_mean_op, feeddict)

                gate_mean_table['train'] = sess.run(gate_mean_val)

                sess.run(tf.local_variables_initializer())  # reset gate_mean
                indices = np.arange(test_size)
                for b in range(0, test_size, batchsize):
                    batch_i = indices[b:b + batchsize]
                    batch_x = dataset.test_X[batch_i]
                    batch_y = dataset.test_Y[batch_i]
                    feeddict = {X: batch_x, Y_Onehot: batch_y}
                    sess.run(gate_mean_op, feeddict)

                gate_mean_table['test'] = sess.run(gate_mean_val)

        results[net] = {'acc':dict()}
        results[net]['acc']['train'] = train_acc_history
        results[net]['acc']['test'] = test_acc_history
        results[net]['gate'] = gate_mean_table

        lines = [
            '\n',net,
            '\nepochs ',str(epochs),
            '\ntrain time ',str(train_time),
            '\ntest time ',str(test_times),
            '\ntrain acc by epoch ',str(train_acc_history),
            '\ntest acc ',str(test_acc_history),
            '\n'
        ]

        with open("experiment1.txt",'a+') as file:
            file.writelines(lines)

    legend = []
    for net in results.keys():
        acc = results[net]['acc']
        for isTrain, history in acc.items():
            plt.plot(range(len(history)),history)
            legend.append(net + ' ' + isTrain)
        gate_mean = results[net]['gate']
        print(net,':',gate_mean)
    plt.gca().legend(legend)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()


