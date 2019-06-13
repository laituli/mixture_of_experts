#import itertools as iter
import time
import functools as func
from mnist import mnist
from cifar import cifar
from omniglot import omniglot
from dataset import combine_datasets, visualize
import numpy as np
np.random.seed(12345)
import tensorflow as tf
tf.random.set_random_seed(12345)
import matplotlib.pyplot as plt
import os
import sys

from datetime import datetime

experiment_name = sys.argv[0].split("/")[-1][:-3]
result_folder = "results/"+experiment_name
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

experiment_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
result_folder+="/"+str(experiment_timestamp)+"/"
os.mkdir(result_folder[:-1])




# DEFINE DATA
image_height = image_width = 32
is_gray = False
original_datasets = [cifar(100)]
#original_datasets = [omniglot]
#original_datasets = [mnist, cifar(10)] # cifar10, 20 or 100 is allowed

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

    #net = '1_layer_gate'
    results = dict()

    def experts(X,Training):
        layer = tf.layers.flatten(X)
        Y = tf.layers.dense(layer, num_classes, tf.nn.softmax)
        return Y

    for num_experts in reversed(range(1,20)):
        layer = tf.layers.flatten(X)
        layer = tf.layers.dense(layer, 64, tf.nn.relu)
        Gate = tf.layers.dense(layer, num_experts, tf.nn.softmax)
        Experts = [experts(X,Training) for i in range(num_experts)]
        Y_Prob = sum([Gate[:,i:i+1]*Expert for i,Expert in enumerate(Experts)])


        # crossentropy
        Task_Loss = -tf.reduce_sum(tf.math.xlogy(Y_Onehot,Y_Prob),axis=1)

        Loss = Task_Loss

        Y_Pred = tf.argmax(Y_Prob,axis=1)
        Acc, Acc_op = tf.metrics.accuracy(Y_True,Y_Pred)
        print(Acc,Acc_op)

        Mnist_gate = tf.boolean_mask(Gate,Is_Mnist)
        Cifar_gate = tf.boolean_mask(Gate,~Is_Mnist)
        gate_mean = {
            'mnist': {i: tf.metrics.mean(Mnist_gate[:, i]) for i in range(num_experts)},
            'cifar': {i: tf.metrics.mean(Cifar_gate[:, i]) for i in range(num_experts)}
        }
        gate_mean_val = {
            'mnist': {i: gate_mean['mnist'][i][0] for i in range(num_experts)},
            'cifar': {i: gate_mean['cifar'][i][0] for i in range(num_experts)}
        }
        gate_mean_op = {
            'mnist': {i: gate_mean['mnist'][i][1] for i in range(num_experts)},
            'cifar': {i: gate_mean['cifar'][i][1] for i in range(num_experts)}
        }
        del gate_mean

        epochs = 20
        batchsize = 128

        Trainstep = tf.train.AdamOptimizer().minimize(Loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
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

                # no earlystop

            end = time.time()
            train_time = end-start-sum(test_times)


            # compute confusion matrix all
            if Gate is not None:
                print("computing confusion matrices")
                confusion_matrix_experts = np.zeros((num_experts, num_classes, num_classes))
                indices = np.arange(test_size)
                for b in range(0, test_size, batchsize):
                    batch_i = indices[b:b + batchsize]
                    batch_x = dataset.test_X[batch_i]
                    batch_y = dataset.test_Y[batch_i]
                    feeddict = {X: batch_x, Y_Onehot: batch_y}
                    y_trues, y_preds, largest_gates = sess.run((Y_True, Y_Pred, tf.argmax(Gate,axis=1)), feeddict)
                    for y_true, y_pred, gate in zip(y_trues, y_preds, largest_gates):
                        confusion_matrix_experts[gate,y_true,y_pred] += 1

                confusion_matrix_all = np.sum(confusion_matrix_experts, axis=0)
                confusion_matrix_all = confusion_matrix_all / np.sum(confusion_matrix_all, axis=1, keepdims=True)
                confusion_matrix_all = confusion_matrix_all * 100 # in %

                print("writing confusion matrices")
                def matrix2rowstrs(matrix, format):
                    rowstrs = []
                    for i, rowvec in enumerate(matrix):
                        rowstr = '\t'.join(map(lambda num: format % num, rowvec))
                        rowstrs.append(rowstr + "\n")
                    return rowstrs

                with open(result_folder + str(num_experts) + "experts_confusion_matrix.txt", 'w') as file:
                    file.write("overall confusion: (rows normalized, in %)\n")
                    file.writelines(matrix2rowstrs(confusion_matrix_all, "%i"))
                    for i, confusion_matrix in enumerate(confusion_matrix_experts):
                        file.write("\n"+str(i)+":th expert's confusion: (integer count)\n")
                        file.writelines(matrix2rowstrs(confusion_matrix, "%i"))

            results[num_experts] = {'acc': dict()}
            results[num_experts]['acc']['train'] = train_acc_history
            results[num_experts]['acc']['test'] = test_acc_history

            gate_mean_table = dict()
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
            results[num_experts]['gate'] = gate_mean_table

        lines = [
            '\nnum experts ',str(num_experts),
            '\nepochs ',str(epochs),
            '\ntrain time ',str(train_time),
            '\ntest time ',str(test_times),
            '\ntrain acc by epoch ',str(train_acc_history),
            '\ntest acc ',str(test_acc_history),
            '\nmax test acc',str(max(test_acc_history)),
            '\n'
        ]

        # with open("experiment1_variation.txt",'a+') as file:
        with open(result_folder+str(num_experts)+"experts.txt",'w') as file:
            file.writelines(lines)

    gate_table_str=""
    for net in results.keys():
        if 'gate' not in results[net]: continue
        gate_mean = results[net]['gate']
        print(net,':',gate_mean)
        for set_name in ["mnist", "cifar"]:

            gate_table_str += str(net) + " " + set_name + " train\n"
            for i in range(net):
                gate_table_str += "%.2f" % gate_mean['train'][set_name][i] + "\t"
            gate_table_str += '\n'

            gate_table_str += str(net) + " " + set_name + " test\n"
            for i in range(net):
                gate_table_str += "%.2f" % gate_mean['test'][set_name][i] + "\t"
            gate_table_str += '\n'

    legend = []
    colors = 'bgrcmy'
    #style = {"train":'','test':'--','max test':'-.'}
    upper = 0
    lower = 1
    vals = {"num_experts":[],"train":[],"test":[],"max test":[]}
    for net in sorted(results.keys()):
        acc = results[net]['acc']

        vals["num_experts"].append(net)

        # train
        val = acc["train"][-1]
        vals["train"].append(val)
        upper = max(upper, val)
        lower = min(lower, val)

        # test
        val = acc["test"][-1]
        vals["test"].append(val)
        lower = min(lower, val)

        # max test
        val = max(acc["test"])
        vals["max test"].append(val)
        upper = max(upper,val)

    for (key, value), color in zip(vals.items(), colors):
        if key == "num_experts": continue
        plt.plot(vals["num_experts"], value, color)
        legend.append(key)
    plt.gca().legend(legend)
    plt.xlabel("number of experts")
    plt.ylabel('accuracy')
    left = min(vals["num_experts"])
    right = max(vals["num_experts"])
    upper = min(1,upper+.05)
    lower = min(upper-.2, lower-0.05)
    plt.axis([left, right, lower, upper])
    plt.savefig(result_folder + 'numE-accuracy.png')
    plt.show()

    with open(result_folder + "gate_table.txt",'w') as file:
        file.write(gate_table_str)


