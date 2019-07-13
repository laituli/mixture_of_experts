import numpy as np
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)

import os
from utils.moe import Gate, Expert, MixtureOfExperts
from utils.train import feed_iter
from utils import plot, save
from utils.file import ModelFile, ResultFile, main_timestamp
from datacode.cifar import cifar100_grouped
import copy

def new_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def mixture_data():
    print("load data")
    from datacode.mnist import mnist
    from datacode.cifar import cifar
    from datacode.dataset import combine_datasets

    dataset = combine_datasets([mnist, cifar(10)], 32, 32, True)
    return dataset


class Experiment:

    def eval(self, try_load=True):
        tf.reset_default_graph()

        print("create model")
        model = MixtureOfExperts(self.x_shape, self.num_classes, self.num_experts,
                                 self.gatef, self.expertf, self.z_size)
        placeholder_tuple = model.X, model.Y_Onehot, model.Z_Onehot

        print("saver")
        saver = tf.train.Saver()
        m_folder = os.path.join("models", "D%s" % self.dataset_name, "E%s" % self.expert_name,
                                "G%s" % self.gate_name, "N%i" % self.num_experts)
        mf = ModelFile(m_folder, timestamp=None if try_load else main_timestamp)

        with new_sess() as sess:
            start_epoch = 0
            to_random_initialize = True
            if try_load:
                ckpt_e = mf.latest_ckpt(self.epochs)
                if ckpt_e is not None:
                    ckpt, start_epoch = ckpt_e
                    to_random_initialize = False
                    print("load model", ckpt)
                    saver.restore(sess, ckpt)

            if to_random_initialize:
                print("random initialization")
                sess.run(tf.global_variables_initializer())

            for e in range(start_epoch, self.epochs):
                print("epoch %i/%i" % (e, self.epochs))

                print("train")
                sess.run(tf.local_variables_initializer())
                for b, feed in enumerate(feed_iter(self.train_tuple, placeholder_tuple,
                                                   datasize=self.trainsize, shuffle=True)):
                    print(end="-", flush=True)
                    sess.run((model.Train_op, model.Acc_op), feed_dict=feed)
                print()
                train_acc = sess.run(model.Acc)

                print("test")
                sess.run(tf.local_variables_initializer())
                for b, feed in enumerate(feed_iter(self.test_tuple, placeholder_tuple,
                                                   datasize=self.testsize, shuffle=False)):
                    print(end="-", flush=True)
                    sess.run(model.Acc_op, feed_dict=feed)
                test_acc = sess.run(model.Acc)
                print()

                print("write")
                with open(mf.train_acc_file(), mode="a") as f:
                    f.write("%i\t%f\n" % (e + 1, train_acc))

                with open(mf.test_acc_file(), mode="a") as f:
                    f.write("%i\t%f\n" % (e + 1, test_acc))

                save_ckpt = e % 5 == 4
                if save_ckpt or e + 1 == self.epochs:
                    ckpt = mf.ckpt(e + 1)
                    saver.save(sess, ckpt)

            print("eval")
            Ops = model.Confusion_op, model.Y_Activation_op, model.Z_Activation_op
            keys = "expert confusion", "overall confusion", "y activation", "z activation"
            Vals = model.ExpertConfusion, model.OverallConfusion, model.Y_Activation, model.Z_Activation
            sess.run(tf.local_variables_initializer())
            for b, feed in enumerate(feed_iter(self.test_tuple, placeholder_tuple,
                                               datasize=self.testsize, shuffle=False)):
                print(end="-", flush=True)
                sess.run(Ops, feed)
            vals = sess.run(Vals)
            print()

            with open(mf.train_acc_file(), "r") as file:
                train_acc = [line.strip().split("\t") for line in file.readlines()]
                train_acc = {int(a): float(b) for a, b in train_acc}
            with open(mf.test_acc_file(), "r") as file:
                test_acc = [line.strip().split("\t") for line in file.readlines()]
                test_acc = {int(a): float(b) for a, b in test_acc}
            result = {key:val for key,val in zip(keys, vals)}
            result["train acc"] = train_acc
            result["test acc"] = test_acc
            return result

class Experiment1(Experiment):

    def __init__(self):
        self.index = 1
        self.name = "Mixture ELinear differentG"
        self.dataset_name = "mixture"
        self.expert_name = "linear"

    def run(self, try_load=True):
        print("run", self.index, self.name)
        self.dataset = mixture_data()
        self.expertf = Expert.linear
        self.gates = {"no-gate": Gate.no_gate,
                 "predesigned": Gate.predesigned,
                 "linear": Gate.linear,
                 "1-hidden": Gate.one_hidden_layer}
        self.epochs = 20
        self.num_experts = 2

        self.trainsize, *self.x_shape = self.dataset.train_X.shape
        self.testsize, self.num_classes = self.dataset.test_Y.shape
        _, self.z_size = self.dataset.train_Z.shape
        self.train_tuple = self.dataset.train_X, self.dataset.train_Y, self.dataset.train_Z
        self.test_tuple = self.dataset.test_X, self.dataset.test_Y, self.dataset.test_Z

        r_folder = os.path.join("result2", self.name)
        rf = ResultFile(r_folder)
        results = dict()

        for c,(self.gate_name, self.gatef) in enumerate(self.gates.items()):
            print("####case",c,self.gate_name,"####")
            result = self.eval(try_load)
            results[self.gate_name] = result
        print("plot")

        plot.epoch_accuracy(results, rf.file("epoch_accuracy"))
        save.confusion(results, rf.t_folder())
        save.activation(results, rf.t_folder())
        for experti in range(2):
            for key in results:
                plot.y_activation(results, key, experti,rf.t_folder(),
                                  groupsize=self.num_classes//self.z_size)


class Experiment2(Experiment):

    def __init__(self):
        self.index = 2
        self.name = "Mixture EConv differentG"
        self.dataset_name = "mixture"
        self.expert_name = "conv"

    def run(self, try_load=True):
        print("run",self.index,self.name)
        self.dataset = mixture_data()
        self.expertf = Expert.convolutional
        self.gates = {"no-gate": Gate.no_gate,
                      "predesigned": Gate.predesigned,
                      "linear": Gate.linear,
                      "1-hidden": Gate.one_hidden_layer}
        self.epochs = 20
        self.num_experts = 2

        self.trainsize, *self.x_shape = self.dataset.train_X.shape
        self.testsize, self.num_classes = self.dataset.test_Y.shape
        _, self.z_size = self.dataset.train_Z.shape
        self.train_tuple = self.dataset.train_X, self.dataset.train_Y, self.dataset.train_Z
        self.test_tuple = self.dataset.test_X, self.dataset.test_Y, self.dataset.test_Z

        results = dict()
        for c, (self.gate_name, self.gatef) in enumerate(self.gates.items()):
            print("####case", c, self.gate_name, "####")
            result = self.eval(try_load)
            results[self.gate_name] = result

        print("plot")
        r_folder = os.path.join("result2", self.name)
        rf = ResultFile(r_folder)
        plot.epoch_accuracy(results, rf.file("epoch_accuracy"))
        save.confusion(results, rf.t_folder())
        save.activation(results, rf.t_folder())
        for experti in range(2):
            for key in results:
                plot.y_activation(results, key, experti,rf.t_folder(),
                                  groupsize=self.num_classes//self.z_size)

class Experiment3(Experiment):

    def __init__(self):
        self.index = 3
        self.name = "Mixture ELinear differentN"
        self.dataset_name = "mixture"
        self.expert_name = "linear"
        self.gate_name = "1-hidden"

    def run(self, try_load=True):
        print("run",self.index,self.name)
        self.dataset = mixture_data()
        self.expertf = Expert.linear
        self.gatef = Gate.one_hidden_layer
        self.epochs = 20
        self.num_experts_s = [20,16,12,10,8,6,4,3,2,1]

        self.trainsize, *self.x_shape = self.dataset.train_X.shape
        self.testsize, self.num_classes = self.dataset.test_Y.shape
        _, self.z_size = self.dataset.train_Z.shape
        self.train_tuple = self.dataset.train_X, self.dataset.train_Y, self.dataset.train_Z
        self.test_tuple = self.dataset.test_X, self.dataset.test_Y, self.dataset.test_Z

        results = dict()
        for c, self.num_experts in enumerate(self.num_experts_s):
            print("####case", self.num_experts, "####")
            result = self.eval(try_load)
            results[self.num_experts] = result
        print("####case predesigned gate####")
        predesigned = copy.copy(self)
        predesigned.gate_name = "predesigned"
        predesigned.gatef = Gate.predesigned
        predesigned.num_experts = predesigned.z_size
        predesigned_result = predesigned.eval(try_load)

        print("plot")
        r_folder = os.path.join("result2", self.name)
        rf = ResultFile(r_folder)
        plot.numE_accuracy(results, rf.file("numE_accuracy"),
                           predesigned_result=predesigned_result,z_size=predesigned.z_size)
        save.confusion(results, rf.t_folder())
        save.activation(results, rf.t_folder())
        for experti in range(4):
            plot.y_activation(results,4,experti,rf.t_folder(),
                              groupsize=self.num_classes//self.z_size)


class Experiment4(Experiment):

    def __init__(self):
        self.index = 4
        self.name = "Cifar100 ELinear differentN"
        self.dataset_name = "cifar100"
        self.expert_name = "linear"
        self.gate_name = "1-hidden"

    def run(self, try_load=True):
        print("run",self.index,self.name)
        self.dataset = cifar100_grouped()
        self.expertf = Expert.linear
        self.gatef = Gate.one_hidden_layer
        self.epochs = 20
        self.num_experts_s = [20,16,12,10,8,6,4,3,2,1]

        self.trainsize, *self.x_shape = self.dataset.train_X.shape
        self.testsize, self.num_classes = self.dataset.test_Y.shape
        _, self.z_size = self.dataset.train_Z.shape
        self.train_tuple = self.dataset.train_X, self.dataset.train_Y, self.dataset.train_Z
        self.test_tuple = self.dataset.test_X, self.dataset.test_Y, self.dataset.test_Z

        results = dict()
        for c, self.num_experts in enumerate(self.num_experts_s):
            print("####case", self.num_experts, "####")
            result = self.eval(try_load)
            results[self.num_experts] = result

        print("####case predesigned gate####")
        predesigned = copy.copy(self)
        predesigned.gate_name = "predesigned"
        predesigned.gatef = Gate.predesigned
        predesigned.num_experts = predesigned.z_size
        predesigned_result = predesigned.eval(try_load)

        print("plot")
        r_folder = os.path.join("result2", self.name)
        rf = ResultFile(r_folder)
        plot.numE_accuracy(results, rf.file("numE_accuracy"),
                           predesigned_result=predesigned_result,z_size=predesigned.z_size)
        save.confusion(results, rf.t_folder())
        save.activation(results, rf.t_folder())
        for experti in range(4):
            plot.y_activation(results,4,experti,rf.t_folder(),
                              groupsize=self.num_classes//self.z_size)


class Experiment5(Experiment):

    def __init__(self):
        self.index = 5
        self.name = "Cifar100 EConv differentN"
        self.dataset_name = "cifar100"
        self.expert_name = "conv"
        self.gate_name = "1-hidden"

    def run(self, try_load=True):
        print("run", self.index, self.name)
        self.dataset = cifar100_grouped()
        self.expertf = Expert.convolutional
        self.gatef = Gate.one_hidden_layer
        self.epochs = 20
        self.num_experts_s = [20, 16, 12, 10, 8, 6, 4, 3, 2, 1]

        self.trainsize, *self.x_shape = self.dataset.train_X.shape
        self.testsize, self.num_classes = self.dataset.test_Y.shape
        _, self.z_size = self.dataset.train_Z.shape
        self.train_tuple = self.dataset.train_X, self.dataset.train_Y, self.dataset.train_Z
        self.test_tuple = self.dataset.test_X, self.dataset.test_Y, self.dataset.test_Z

        results = dict()
        for c, self.num_experts in enumerate(self.num_experts_s):
            print("####case", self.num_experts, "####")
            result = self.eval(try_load)
            results[self.num_experts] = result

        print("####case predesigned gate####")
        predesigned = copy.copy(self)
        predesigned.gate_name = "predesigned"
        predesigned.gatef = Gate.predesigned
        predesigned.num_experts = predesigned.z_size
        predesigned_result = predesigned.eval(try_load)

        print("plot")
        r_folder = os.path.join("result2", self.name)
        rf = ResultFile(r_folder)
        plot.numE_accuracy(results, rf.file("numE_accuracy"),
                           predesigned_result=predesigned_result,z_size=predesigned.z_size)
        save.confusion(results, rf.t_folder())
        save.activation(results, rf.t_folder())
        for experti in range(4):
            plot.y_activation(results, 4, experti, rf.t_folder(),
                              groupsize=self.num_classes // self.z_size)


#Experiment1().run(try_load=True)
#Experiment2().run(try_load=True)
Experiment3().run(try_load=True)
Experiment4().run(try_load=True)
#Experiment5().run(try_load=True)
#Experiment3().run(try_load=True)
#Experiment3().run(try_load=True)
