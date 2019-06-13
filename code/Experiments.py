from shared import *
from dataset import combine_datasets
import os
from datetime import datetime

import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)

def mixture_data():
    print("load data")
    from mnist import mnist
    from cifar import cifar
    dataset = combine_datasets([mnist, cifar(10)], 32, 32, True)
    return dataset

def cifar100():
    print("load data")
    from cifar import cifar
    return cifar(100)

def experiment_gatef_loop(all_gatef, dataset,
                          epochs, num_experts, expertf,
                          folder):
    trainsize, *x_shape = dataset.train_X.shape
    testsize, num_classes = dataset.test_Y.shape

    results = {"acc":{}, "confusion":{}, "activation":{}}
    for name, gatef in all_gatef.items():
        print("####case",name,"####")
        with tf.Graph().as_default() as graph:
            X = tf.placeholder(tf.float32, [None]+x_shape)
            Y_Onehot = tf.placeholder(tf.float32, (None, num_classes))
            model = Mixture_of_experts(X, Y_Onehot,
                                       num_experts, num_classes,
                                       gatef, expertf)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())

                trainiterf = lambda : feed_iter((dataset.train_X,dataset.train_Y),
                                                 (X,Y_Onehot),
                                                 trainsize, 128, True)
                testiterf = lambda : feed_iter((dataset.test_X,dataset.test_Y),
                                                 (X,Y_Onehot),
                                                 testsize, 128, False)
                acc, acc_op = tf.metrics.accuracy(model.Y_True,model.Y_Pred)
                train_acc, test_acc = train(sess, epochs, trainiterf, testiterf,
                      model.Train_op, acc_op, acc)
                results["acc"][name] = {"train":train_acc,"test":test_acc}
                results["confusion"][name] = confusion_matrices(sess, testiterf(),
                                                       model.Y_True, model.Y_Pred, model.Largest_gate,
                                                       num_experts, num_classes, folder + name + "_confusion.txt")

                results["activation"][name] = activation_matrix(sess, testiterf(),
                                                       model.SuperLabel,model.Gate,
                                                       2, num_experts, folder + name + "_activation.txt")

    plot_epoch_accuracy(results["acc"], folder+"epoch_accuracy.png")
    return results


def experiment_numE_loop(all_num_experts, dataset, epochs,
                         gatef, expertf, folder):
    trainsize, *x_shape = dataset.train_X.shape
    testsize, num_classes = dataset.test_Y.shape

    results = {"acc": {}, "confusion": {}, "activation": {}}
    for num_experts in all_num_experts:
        print("####case", num_experts, "####")
        with tf.Graph().as_default() as graph:
            X = tf.placeholder(tf.float32, [None]+x_shape)
            Y_Onehot = tf.placeholder(tf.float32, (None, num_classes))
            model = Mixture_of_experts(X, Y_Onehot,
                                       num_experts, num_classes,
                                       gatef, expertf)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())

                trainiterf = lambda: feed_iter((dataset.train_X, dataset.train_Y),
                                               (X, Y_Onehot),
                                               trainsize, 128, True)
                testiterf = lambda: feed_iter((dataset.test_X, dataset.test_Y),
                                              (X, Y_Onehot),
                                              testsize, 128, False)
                acc, acc_op = tf.metrics.accuracy(model.Y_True, model.Y_Pred)

                train_acc, test_acc = train(sess, epochs, trainiterf, testiterf,
                                            model.Train_op, acc_op, acc)

                results["acc"][num_experts] = {"train": train_acc, "test": test_acc}
                results["confusion"][num_experts] = confusion_matrices(sess, testiterf(),
                                                       model.Y_True, model.Y_Pred, model.Largest_gate,
                                                       num_experts, num_classes, folder + "%iexperts_confusion.txt" % num_experts)
                results["activation"][num_experts] = activation_matrix(sess, testiterf(),
                                                       model.SuperLabel, model.Gate,
                                                       num_classes//10, num_experts, folder + "%i_activation.txt" % num_experts)
    plot_numE_accuracy(results["acc"], folder + "numE_accuracy.png")
    return results


def Experiment1(folder):
    dataset = mixture_data()
    expertf = Expert.linear
    gates = {"no-gate": Gate.no_gate,
             "predesigned": Gate.predesigned,
             "linear": Gate.linear,
             "1-hidden": Gate.one_hidden_layer}
    epochs = 20
    num_experts = 2

    experiment_gatef_loop(gates, dataset, epochs, num_experts, expertf, folder)

def Experiment2(folder):
    dataset = mixture_data()
    expertf = Expert.convolutional
    gates = {"no-gate": Gate.no_gate,
             "predesigned": Gate.predesigned,
             "linear": Gate.linear,
             "1-hidden": Gate.one_hidden_layer}
    epochs = 20
    num_experts = 2

    experiment_gatef_loop(gates, dataset, epochs, num_experts, expertf, folder)


def Experiment3(folder):
    dataset = mixture_data()
    expertf = Expert.linear
    gatef = Gate.one_hidden_layer
    all_num_experts = [20,16,12,10,8,6,4,3,2,1]
    epochs = 20

    experiment_numE_loop(all_num_experts, dataset, epochs, gatef, expertf, folder)

def Experiment4(folder):
    dataset = cifar100()
    expertf = Expert.linear
    gatef = Gate.one_hidden_layer
    all_num_experts = [20,16,12,10,8,6,4,3,2,1]
    epochs = 20

    experiment_numE_loop(all_num_experts, dataset, epochs, gatef, expertf, folder)


def Experiment5(folder):
    dataset = cifar100()
    expertf = Expert.convolutional
    gatef = Gate.one_hidden_layer
    all_num_experts = [10,8,6,4,3,2,1]
    epochs = 20

    experiment_numE_loop(all_num_experts, dataset, epochs, gatef, expertf, folder)



folder = "results/Experiments/"
if not os.path.isdir(folder):
    os.mkdir(folder)
experiment_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#folder +=str(experiment_timestamp)+"/"
#os.mkdir(folder)

names = "Mixture_Elinear", "Mixture_ECNN", "Mixture_ELinear_Mgrow", "Cifar100_ELinear_Mgrow", "Cifar100_ECNN_Mgrow"
fs = Experiment1, Experiment2, Experiment3, Experiment4, Experiment5
l = list(zip(names,fs))
for i in [4]:
    name, f = l[i]
    i_folder = folder+name+"/"
    if not os.path.isdir(i_folder):
        os.mkdir(i_folder)
    i_folder += str(experiment_timestamp)+"/"
    os.mkdir(i_folder)

    print("########")
    print("experiment:", name)
    f(i_folder)
    print("########")
print("all executed")
