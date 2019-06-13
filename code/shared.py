import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Expert:

    @staticmethod
    def linear(X, num_classes, activation=tf.nn.softmax):
        layer = tf.layers.flatten(X)
        Y = tf.layers.dense(layer,num_classes,tf.nn.softmax)
        return Y

    @staticmethod
    def convolutional(X, num_classes, activation=tf.nn.softmax):
        layer = X
        layer = tf.layers.conv2d(layer,64,(3,3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer,64,(3,3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.flatten(layer)
        layer = tf.layers.dense(layer, 64, tf.nn.relu)
        Y = tf.layers.dense(layer, num_classes, tf.nn.softmax)
        return Y


class Gate:

    @staticmethod
    def no_gate(X, num_experts, **kwargs):
        """

        :param X:
        :param num_experts:
        :param broadcast:
        :return:
                [[1 0 0 0]] if broadcast
                else to shape (X.len, num_experts)
        """
        gate = tf.eye(1,num_experts)
        return tf.tile(gate,tf.stack((tf.shape(X)[0],1)))

    @staticmethod
    def predesigned(X, num_experts, belong, **kwargs):
        """

        :param X:
        :param num_experts:
        :param belong: idx of expert for each X
        :return: one hot version of idx
        """
        r = tf.range(num_experts)
        cross_equal = tf.equal(belong[:,tf.newaxis],r[tf.newaxis])
        onehot = tf.cast(cross_equal,tf.float32)
        return onehot

    @staticmethod
    def linear(X, num_experts, activation=tf.nn.softmax, **kwargs):
        layer = tf.layers.flatten(X)
        G = tf.layers.dense(layer, num_experts, activation=activation)
        return G

    @staticmethod
    def one_hidden_layer(X, num_experts, activation=tf.nn.softmax, **kwargs):
        layer = tf.layers.flatten(X)
        layer = tf.layers.dense(layer,64,activation=tf.nn.relu)
        G = tf.layers.dense(layer, num_experts, activation=activation)
        return G


def matrix_strings(matrix, cell_format="%i\t"):
    strs = []
    for i, row in enumerate(matrix):
        for cell in row:
            strs.append(cell_format%cell)
        strs.append("\n")
    return strs

def feed_iter(data_tuple, place_holder_tuple, datasize, batchsize=128, shuffle=True, mapping=lambda x:x):
    indices = np.random.permutation(datasize) if shuffle else np.arange(datasize)
    for b in range(0, datasize, batchsize):
        b_idx = indices[b:b+batchsize]
        b_datatuple = tuple(data[b_idx] for data in data_tuple)
        b_datatuple = mapping(b_datatuple)
        feeddict = dict(zip(place_holder_tuple, b_datatuple))
        yield feeddict

def confusion_matrices(sess, feed_iter,
                       tf_true, tf_pred, tf_largest_gate,
                       num_experts, num_classes, outfile):
    """

    :param sess:
    :param feed_iter:
    :param tf_true:
    :param tf_pred:
    :param tf_largest_gate:
    :param num_experts:
    :param num_classes:
    :return: count {"overall":np[true,pred], "expert":np[i,true,pred], "overall_percent":?}
    """
    print("compute confusion")

    confusion_matrix_experts = np.zeros((num_experts, num_classes, num_classes))
    for feeddict in feed_iter:
        y_trues, y_preds, largest_gates = sess.run((tf_true, tf_pred, tf_largest_gate), feeddict)
        for y_true, y_pred, gate in zip(y_trues, y_preds, largest_gates):
            confusion_matrix_experts[gate, y_true, y_pred] += 1

    confusion_matrix_all = np.sum(confusion_matrix_experts, axis=0)
    confusion_matrix_all_percent = 100 * confusion_matrix_all / np.sum(confusion_matrix_all, axis=1, keepdims=True)

    with open(outfile, 'w') as file:
        file.write("overall confusion: (integer count)\n")
        file.writelines(matrix_strings(confusion_matrix_all))
        file.write("overall confusion percent: (rows normalized in %)\n")
        file.writelines(matrix_strings(confusion_matrix_all_percent))

        for i, confusion_matrix in enumerate(confusion_matrix_experts):
            file.write("\n" + str(i) + ":th expert's confusion: (integer count)\n")
            file.writelines(matrix_strings(confusion_matrix))

    return {"overall":confusion_matrix_all,
            "overall_percent":confusion_matrix_all_percent,
            "experts":confusion_matrix_experts}

def activation_matrix(sess, feed_iter,
                      tf_super_label, tf_gate,
                      num_super_labels, num_experts, outfile):

    print("compute activation")

    gate_sum = np.zeros((num_super_labels, num_experts))
    label_size = np.zeros(num_super_labels)

    for feeddict in feed_iter:
        b_super_label, b_gate = sess.run((tf_super_label, tf_gate),feeddict)
        bool_where = np.equal(  # (s,1) == (1,b) -> (s,b)
            np.arange(num_super_labels)[:, np.newaxis],
            b_super_label[np.newaxis, :])


        for super_label in range(num_super_labels):
            gate_of_label = b_gate[bool_where[super_label]]
            gate_sum[super_label] += np.sum(gate_of_label, axis=0)
        label_size += np.sum(bool_where.astype(np.int32), axis=1)

    activation = gate_sum / label_size[:,np.newaxis]

    with open(outfile,"w") as file:
        file.write("row indicate super label, column indicate expert\n")
        file.writelines(matrix_strings(activation,cell_format="%.2f\t"))

    return activation

def train(sess, epochs, feed_iter_f, val_iter_f,
          trainstep,
          measure_op, measure,
          measure_reset_op=tf.local_variables_initializer):

    print("start train loop")

    train_measure_history = []
    valid_measure_history = []
    for e in range(epochs):
        print(e,"/",epochs)
        sess.run(measure_reset_op())
        print("train_set")
        for b, feed in enumerate(feed_iter_f()):
            if b % 128 == 127:
                print()
            print(end="-",flush=True)
            sess.run((trainstep, measure_op), feed)
        train_measure_history.append(sess.run(measure))
        print()
        print("validation_set")
        sess.run(measure_reset_op())
        for feed in val_iter_f():
            sess.run(measure_op, feed)
        valid_measure_history.append(sess.run(measure))
    return train_measure_history, valid_measure_history

def plot_epoch_accuracy(acc, outfile):
    print("plotting accuracy")
    legend = []
    colors = 'bgrcmy'
    styles = {"train":'','test':'--'}
    max_x = 0
    max_y = 0
    for modelname, color in zip(acc, colors):
        modelacc = acc[modelname]
        for is_train, acc_history in modelacc.items():
            x = range(len(acc_history))
            max_x = max(max_x,max(x))
            y = acc_history
            max_y = max(max_y,max(y))
            style = styles[is_train]
            plt.plot(x, y, color + style)
            legend.append(modelname+" "+is_train)

    max_y = min(max_y+.05,1)

    plt.gca().legend(legend)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.axis([0,max_x, max_y-0.25, max_y])
    plt.savefig(outfile)
    plt.show()

def plot_numE_accuracy(acc, outfile):
    print("plotting accuracy")
    legend = []
    colors = 'bgrcmy'

    max_x = max(acc)
    num_e = sorted(acc)
    acc = {"train end":[acc[numE]["train"][-1] for numE in sorted(acc)],
           "test end":[acc[numE]["test"][-1] for numE in sorted(acc)],
           "test max":[max(acc[numE]["test"]) for numE in sorted(acc)]}

    all_y = [a for d in acc.values() for a in d]
    max_y = max(all_y)
    min_y = min(all_y)
    max_y = min(max_y+.05,1)
    min_y = max(min_y-.05,0)
    for (key, history), color in zip(acc.items(),colors):
        plt.plot(num_e, history, color)
        legend.append(key)
    plt.gca().legend(legend)
    plt.xlabel('number of expert')
    plt.ylabel('accuracy')
    plt.axis([0, max_x, min_y, max_y])
    plt.savefig(outfile)
    plt.show()


class Mixture_of_experts:

    def __init__(self,X_img, Y_onehot, num_experts, num_classes, gatef, expertf):
        self.X = X_img
        self.Y_Onehot = Y_onehot
        self.Y_True = tf.argmax(Y_onehot, axis=1, output_type=tf.int32)

        # irrelevant if is not mixture dataset
        self.SuperLabel = self.Y_True//10
        kwargs = {"belong":self.SuperLabel} # on predesigned gate

        self.Gate = gatef(X_img, num_experts, **kwargs)
        self.Largest_gate = tf.argmax(self.Gate, axis=1, output_type=tf.int32)
        self.Experts = [expertf(X_img, num_classes) for _ in range(num_experts)]
        self.Y_Prob = sum([self.Gate[:,i:i+1]*Expert for i,Expert in enumerate(self.Experts)])
        self.Y_Pred = tf.argmax(self.Y_Prob, axis=1, output_type=tf.int32)
        self.Loss = -tf.reduce_sum(tf.math.xlogy(self.Y_Onehot,self.Y_Prob),axis=1)
        self.Train_op = tf.train.AdamOptimizer().minimize(self.Loss)
