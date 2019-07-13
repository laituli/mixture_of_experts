import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)

import os
from datetime import datetime

from utils import plot, compute, save, train
from utils.train import *
from utils.moe import *


def force_mkdir(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)

def mixture_data():
    print("load data")
    from datacode.mnist import mnist
    from datacode.cifar import cifar
    from datacode.dataset import combine_datasets

    dataset = combine_datasets([mnist, cifar(10)], 32, 32, True)
    return dataset

def cifar100():
    print("load data")
    from datacode.cifar import cifar
    return cifar(100)

def cifar100and20():
    print("load data")
    from datacode.cifar import cifar
    cifar100 = cifar(100)
    cifar20 = cifar(20)
    cifar100.train_Z, cifar100.test_Z = cifar20.train_Y, cifar20.test_Y
    return cifar100

class Experiment1:
    index = 1
    name = "Mixture ELinear differentG"

    @staticmethod
    def run(folder):
        dataset = mixture_data()
        expertf = Expert.linear
        gates = {"no-gate": Gate.no_gate,
                 "predesigned": Gate.predesigned,
                 "linear": Gate.linear,
                 "1-hidden": Gate.one_hidden_layer}
        epochs = 20
        num_experts = 2
        result = experiment_gatef_loop(gates, dataset, epochs, num_experts, expertf, folder)

        for gname, activation in result["class activation"].items():
            for i in [0, 1]:
                plot.plot_activation_of_expert(activation, i, folder + gname + " expert %i activation.png" % i)

class Experiment2:
    index = 2
    name = "Mixture EConv differentG"

    @staticmethod
    def run(folder):
        dataset = mixture_data()
        expertf = Expert.convolutional
        gates = {"no-gate": Gate.no_gate,
                 "predesigned": Gate.predesigned,
                 "linear": Gate.linear,
                 "1-hidden": Gate.one_hidden_layer}
        epochs = 20
        num_experts = 2
        result = experiment_gatef_loop(gates, dataset, epochs, num_experts, expertf, folder)

        for gname, activation in result["class activation"].items():
            for i in [0, 1]:
                plot.plot_activation_of_expert(activation, i, folder + gname + " expert %i activation.png" % i)

class Experiment3:
    index = 3
    name = "Mixture ELinear different#E"

    @staticmethod
    def run(folder):
        dataset = mixture_data()

        expertf = Expert.linear
        gatef = Gate.one_hidden_layer
        all_num_experts = [20, 16, 12, 10, 8, 6, 4, 3, 2, 1]
        epochs = 20
        z_size = 2

        results = experiment_numE_loop(all_num_experts, dataset, epochs, gatef, expertf, folder)
        result2 = experiment_gatef_loop({"predesigned": Gate.predesigned}, dataset, epochs, z_size, expertf, folder)
        plot.special_num_E_accuracy(results["acc"],result2["acc"]["predesigned"], z_size,
                                    folder + "numE_accuracy_with_additional_predesigned_gate.png")

        for number_of_experts, activation in results["class activation"].items():
            for i in range(number_of_experts):
                print("plot activation",i,"/",number_of_experts)
                plot.plot_activation_of_expert(activation, i,
                                               folder + " expert %i-%i activation.png" % (i,number_of_experts))

class Experiment4:
    index = 4
    name = "Cifar100 ELinear different#E"

    @staticmethod
    def run(folder):
        dataset = cifar100and20()

        expertf = Expert.linear
        gatef = Gate.one_hidden_layer
        all_num_experts = [20, 16, 12, 10, 8, 6, 4, 3, 2, 1]
        epochs = 20
        z_size = 20

        results = experiment_numE_loop(all_num_experts, dataset, epochs, gatef, expertf, folder)
        result2 = experiment_gatef_loop({"predesigned": Gate.predesigned}, dataset, epochs, z_size, expertf, folder)
        plot.special_num_E_accuracy(results["acc"], result2["acc"]["predesigned"], z_size,
                                    folder + "numE_accuracy_with_additional_predesigned_gate.png")

        for number_of_experts, activation in results["class activation"].items():
            for i in range(number_of_experts):
                print("plot activation",i,"/",number_of_experts)
                plot.plot_activation_of_expert(activation, i,
                                               folder + " expert %i-%i activation.png" % (i,number_of_experts))

class Experiment5:
    index = 5
    name = "Cifar100 EConv different#E"

    @staticmethod
    def run(folder):
        dataset = cifar100and20()
        expertf = Expert.convolutional
        gatef = Gate.one_hidden_layer
        all_num_experts = [20, 16, 12, 10, 8, 6, 4, 3, 2, 1]
        epochs = 20
        z_size = 20

        results = experiment_numE_loop(all_num_experts, dataset, epochs, gatef, expertf, folder)
        result2 = experiment_gatef_loop({"predesigned": Gate.predesigned}, dataset, epochs, z_size, expertf, folder)
        plot.special_num_E_accuracy(results["acc"],result2["acc"]["predesigned"], z_size,
                                    folder + "numE_accuracy_with_additional_predesigned_gate.png")

        for number_of_experts, activation in results["class activation"].items():
            for i in range(number_of_experts):
                print("plot activation",i,"/",number_of_experts)
                plot.plot_activation_of_expert(activation, i,
                                               folder + " expert %i-%i activation.png" % (i,number_of_experts))

class Experiment6:
    index = 6
    name = "Mixture EConv different#E"

    @staticmethod
    def run(folder):
        dataset = mixture_data()
        expertf = Expert.convolutional
        gatef = Gate.one_hidden_layer
        all_num_experts = [20, 16, 12, 10, 8, 6, 4, 3, 2, 1]
        epochs = 20
        z_size = 2

        result2 = experiment_gatef_loop({"predesigned": Gate.predesigned}, dataset, epochs, z_size, expertf, folder)
        results = experiment_numE_loop(all_num_experts, dataset, epochs, gatef, expertf, folder)
        plot.special_num_E_accuracy(results["acc"],result2["acc"]["predesigned"],z_size,
                                    folder + "numE_accuracy_with_additional_predesigned_gate.png")

        for number_of_experts, activation in results["class activation"].items():
            for i in range(number_of_experts):
                print("plot activation",i,"/",number_of_experts)
                plot.plot_activation_of_expert(activation, i,
                                               folder + " expert %i-%i activation.png" % (i,number_of_experts))

class Experiment7:
    index= 7
    name = "Mixture EHetero differentG"

    @staticmethod
    def run(folder):
        dataset = mixture_data()
        expertsf = [Expert.linear, Expert.convolutional]
        gates = {"predesigned": Gate.predesigned,
                 "linear": Gate.linear,
                 "1-hidden": Gate.one_hidden_layer}
        num_experts = len(expertsf)
        epochs = 20

        result = experiment_gatef_loop(gates, dataset, epochs, num_experts, expertsf, folder)

        for gname, activation in result["class activation"].items():
            for i in [0, 1]:
                plot.plot_activation_of_expert(activation, i, folder + gname + " expert %i activation.png" % i)


def experiment_gatef_loop(all_gatef, dataset,
                          epochs, num_experts, expertf,
                          folder):
    trainsize, *x_shape = dataset.train_X.shape
    testsize, num_classes = dataset.test_Y.shape

    hasZ = hasattr(dataset,"train_Z")
    if hasZ: _, z_size = dataset.train_Z.shape
    else: z_size = None

    results = {"acc":{}, "confusion":{}, "set activation":{}, "class activation":{}}
    for name, gatef in all_gatef.items():
        print("####case",name,"####")
        with tf.Graph().as_default() as graph:
            X = tf.placeholder(tf.float32, [None]+x_shape)
            Y_Onehot = tf.placeholder(tf.float32, (None, num_classes))

            if hasZ: Z_Onehot = tf.placeholder(tf.float32, (None, z_size))
            else: Z_Onehot = None

            model = Mixture_of_experts(X, Y_Onehot,
                                       num_experts, num_classes,
                                       gatef, expertf, Z_Onehot=Z_Onehot)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())

                traintuple = dataset.traintuple(hasZ)
                testtuple = dataset.testtuple(hasZ)
                if hasZ: tensortuple = X, Y_Onehot, Z_Onehot
                else: tensortuple = X, Y_Onehot

                def trainiterf():
                    return feed_iter(traintuple, tensortuple,
                                     trainsize, 128, shuffle=True)
                def testiterf():
                    return feed_iter(testtuple, tensortuple,
                                     testsize, 128, shuffle=False)

                acc, acc_op = tf.metrics.accuracy(model.Y_True,model.Y_Pred)
                train_acc, test_acc = train(
                    sess, epochs,
                    trainiterf, testiterf,
                    model.Train_op, acc_op, acc)

                results["acc"][name] = {"train":train_acc,"test":test_acc}
                results["confusion"][name] = compute.confusion_matrices(sess, testiterf(),
                                                       model.Y_True, model.Y_Pred, model.Largest_gate,
                                                       num_experts, num_classes)

                save.save_confusion(results["confusion"][name], folder + name + "_confusion.txt")

                if model.Z_True is not None:
                    results["set activation"][name] = compute.activation_matrix(
                        sess, testiterf(), model.Z_True,model.Gate,z_size, num_experts)
                    save.save_activation(results["set activation"][name], folder + name + "_set_activation.txt")

                results["class activation"][name] = compute.activation_matrix(
                    sess, testiterf(),model.Y_True, model.Gate, num_classes, num_experts)
                save.save_activation(results["class activation"][name], folder + name + "_class_activation.txt")

    plot.plot_epoch_accuracy(results["acc"], folder+"epoch_accuracy.png")
    return results


def experiment_numE_loop(all_num_experts, dataset, epochs,
                         gatef, expertf, folder):
    trainsize, *x_shape = dataset.train_X.shape
    testsize, num_classes = dataset.test_Y.shape

    hasZ = hasattr(dataset,"train_Z")

    if hasZ: _, z_size = dataset.train_Z.shape
    else: z_size = None

    results = {"acc": {}, "confusion": {}, "set activation": {}, "class activation":{}}
    for num_experts in all_num_experts:
        print("####case", num_experts, "####")
        with tf.Graph().as_default() as graph:
            X = tf.placeholder(tf.float32, [None]+x_shape)
            Y_Onehot = tf.placeholder(tf.float32, (None, num_classes))

            if not hasZ: Z_Onehot = None
            else: Z_Onehot = tf.placeholder(tf.float32, (None, z_size))

            model = Mixture_of_experts(X, Y_Onehot,
                                       num_experts, num_classes,
                                       gatef, expertf, Z_Onehot=Z_Onehot)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())

                if hasZ:
                    traintuple = dataset.train_X, dataset.train_Y,dataset.train_Z
                    testtuple = dataset.test_X, dataset.test_Y, dataset.test_Z
                    tensortuple = X, Y_Onehot, Z_Onehot
                else:
                    traintuple = dataset.train_X, dataset.train_Y
                    testtuple = dataset.test_X, dataset.test_Y
                    tensortuple = X, Y_Onehot

                def trainiterf():
                    return feed_iter(traintuple, tensortuple,
                                     trainsize, 128, shuffle=True)
                def testiterf():
                    return feed_iter(testtuple, tensortuple,
                                     testsize, 128, shuffle=False)

                acc, acc_op = tf.metrics.accuracy(model.Y_True, model.Y_Pred)
                train_acc, test_acc = train(sess, epochs, trainiterf, testiterf,
                                            model.Train_op, acc_op, acc)

                results["acc"][num_experts] = {"train": train_acc, "test": test_acc}
                results["confusion"][num_experts] = compute.confusion_matrices(
                    sess, testiterf(),
                    model.Y_True, model.Y_Pred, model.Largest_gate,
                    num_experts, num_classes)
                save.save_confusion(results["confusion"][num_experts],
                                    folder + "%iexperts_confusion.txt" % num_experts)

                if hasZ:
                    results["set activation"][num_experts] = compute.activation_matrix(
                        sess, testiterf(),
                        model.Z_True, model.Gate,
                        z_size, num_experts)
                    save.save_activation(results["set activation"][num_experts],
                                         folder + "%i_set_activation.txt" % num_experts)

                results["class activation"][num_experts] = compute.activation_matrix(
                    sess, testiterf(),model.Y_True, model.Gate, num_classes, num_experts)
                save.save_activation(results["class activation"][num_experts], folder + "%i_class_activation.txt" % num_experts)


    plot.plot_numE_accuracy(results["acc"], folder + "numE_accuracy.png")
    return results



folder = "result/"
print("make folder if not exist:", folder)
force_mkdir(folder)
experiment_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

#folder +=str(experiment_timestamp)+"/"
#force_mkdir(folder)

experiments = Experiment1, Experiment2, Experiment3, Experiment4, Experiment5, Experiment6, Experiment7
experiments = {e.index : e for e in experiments}

for i in [3,4,7]:
    experiment = experiments[i]
    i_folder = folder + experiment.name + "/"
    print("make folder if not exist:",i_folder)
    force_mkdir(i_folder)
    i_folder += str(experiment_timestamp) + "/"
    print("make folder if not exist:", i_folder)
    force_mkdir(i_folder)

    print("###############")
    print("start running experiment",i,":",experiment.name)
    experiment.run(i_folder)
    print("finishing experiment", i, ":", experiment.name)
    print("###############")

print("all executed")


"""

folder = "testfolder/"
# folder = "results/Experiments/"
force_mkdir(folder)
experiment_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

#folder +=str(experiment_timestamp)+"/"
#force_mkdir(folder)

experiments = Experiment5, Experiment7
experiments = {e.index : e for e in experiments}

for i in sorted(experiments):
    experiment = experiments[i]
    i_folder = folder + experiment.name + "/"
    force_mkdir(i_folder)
    i_folder += str(experiment_timestamp) + "/"

    print("###############")
    print("start running experiment",i,":",experiment.name)
    experiment.run(i_folder)
    print("finishing experiment", i, ":", experiment.name)
    print("###############")

print("all executed")
exit()
"""