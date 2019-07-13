import tensorflow as tf

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

    @staticmethod
    def convolutional_complex(X, num_classes, activation=tf.nn.softmax):
        layer = X
        layer = tf.layers.conv2d(layer, 128, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer, 128, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer, 128, (3, 3), padding='same', activation=tf.nn.relu)
        layer = tf.layers.conv2d(layer, 128, (3, 3), padding='same', activation=tf.nn.relu)
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
    def predesigned(Z_Onehot, **kwargs):
        """

        :param X:
        :param num_experts:
        :param belong: idx of expert for each X
        :return: one hot version of idx
        """
        """
        r = tf.range(num_experts)
        cross_equal = tf.equal(belong[:,tf.newaxis],r[tf.newaxis])
        onehot = tf.cast(cross_equal,tf.float32)
        """

        return Z_Onehot

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



class MixtureOfExperts:

    def __init__(self, xshape, num_classes, num_experts, gatef, expertf, z_size):
        self.xshape = xshape
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.gatef = gatef
        self.expertf = expertf
        self.z_size = z_size


        self._generate_placeholder()
        self._generate_computation()
        self._generate_evaluation()

    def _generate_placeholder(self):
        self.X = tf.placeholder(tf.float32, [None] + self.xshape)
        self.Y_Onehot = tf.placeholder(tf.float32, (None, self.num_classes))
        self.Z_Onehot = tf.placeholder(tf.float32, (None, self.z_size))

    def _generate_computation(self):
        self.Gate = self.gatef(X=self.X, num_experts=self.num_experts, Z_Onehot=self.Z_Onehot)
        if isinstance(self.expertf, list):
            self.Experts = [ef(self.X, self.num_classes) for ef in self.expertf]
        else:
            self.Experts = [self.expertf(self.X, self.num_classes) for _ in range(self.num_experts)]
        self.WeightedExperts = [self.Gate[:, i:i + 1] * Expert for i, Expert in enumerate(self.Experts)]
        self.Y_Prob = sum(self.WeightedExperts)
        self.Loss = -tf.reduce_sum(tf.math.xlogy(self.Y_Onehot, self.Y_Prob), axis=1)
        self.Train_op = tf.train.AdamOptimizer().minimize(self.Loss)

    def _generate_evaluation(self):
        self.Y_True = tf.argmax(self.Y_Onehot, axis=1, output_type=tf.int32)
        self.Y_Pred = tf.argmax(self.Y_Prob, axis=1, output_type=tf.int32)
        self.Y_HotPred = tf.one_hot(self.Y_Pred, self.num_classes)
        self.Z_True = tf.argmax(self.Z_Onehot, axis=1, output_type=tf.int32)
        self.G_Pred = tf.argmax(self.Gate, axis=1, output_type=tf.int32)
        self.G_Onehot = tf.one_hot(self.G_Pred,self.num_experts)


        self.Acc, self.Acc_op = tf.metrics.accuracy(self.Y_True, self.Y_Pred)
        self._generate_confusion()
        self._generate_activation()

    def _generate_confusion(self):
        self.ExpertConfusion = tf.Variable(
            initial_value=tf.zeros([self.num_experts, self.num_classes, self.num_classes]),
            collections=[tf.GraphKeys.LOCAL_VARIABLES])
        self.OverallConfusion = tf.reduce_sum(self.ExpertConfusion,axis=0)

        # (B,G) x (B,Y) x (B,Y)
        # (B,G,1,1) x (B,1,Y,1) x (B,1,1,Y)
        # (B,G,Y,Y)
        # (G,Y,Y)
        G_Onehot = self.G_Onehot[:,:,tf.newaxis,tf.newaxis]
        Y_Onehot = self.Y_Onehot[:,tf.newaxis,:,tf.newaxis]
        Y_HotPred = self.Y_HotPred[:,tf.newaxis,tf.newaxis,:]
        SampleConfusion = G_Onehot * Y_Onehot * Y_HotPred
        BatchConfusion = tf.reduce_sum(SampleConfusion,axis=0)
        self.Confusion_op = tf.assign_add(self.ExpertConfusion, BatchConfusion)

    def _generate_activation(self):
        Y_GateSum = tf.Variable(
            initial_value=tf.zeros([self.num_classes, self.num_experts]),
            collections = [tf.GraphKeys.LOCAL_VARIABLES])
        Y_Count = tf.Variable(
            initial_value=tf.zeros([self.num_classes]),
            collections=[tf.GraphKeys.LOCAL_VARIABLES])
        self.Y_Activation = Y_GateSum / Y_Count[:,tf.newaxis]
        Z_GateSum = tf.Variable(
            initial_value=tf.zeros([self.z_size, self.num_experts]),
            collections = [tf.GraphKeys.LOCAL_VARIABLES])
        Z_Count = tf.Variable(
            initial_value=tf.zeros([self.z_size]),
            collections=[tf.GraphKeys.LOCAL_VARIABLES])
        self.Z_Activation = Z_GateSum / Z_Count[:,tf.newaxis]

        Gate = self.Gate[:, tf.newaxis, :]
        # Y activation
        # (B,Y) x (B,G)
        # (B,Y,1) x (B,1,G)
        # (B,Y,G)
        # (Y,G)
        Y_OneHot = self.Y_Onehot[:,:,tf.newaxis]
        Sample_Y_Gate = Y_OneHot * Gate
        Batch_Y_Gate = tf.reduce_sum(Sample_Y_Gate,axis=0)
        Batch_Y_Count = tf.reduce_sum(self.Y_Onehot,axis=0)
        Y_Count_op = tf.assign_add(Y_Count,Batch_Y_Count)
        with tf.control_dependencies([Y_Count_op]):
            self.Y_Activation_op = tf.assign_add(Y_GateSum, Batch_Y_Gate)

        # Z activation
        # (B,Z) x (B,G)
        # (B,Z,1) x (B,1,G)
        # (B,Z,G)
        # (Z,G)
        Z_OneHot = self.Z_Onehot[:, :, tf.newaxis]
        Sample_Z_Gate = Z_OneHot * Gate
        Batch_Z_Gate = tf.reduce_sum(Sample_Z_Gate, axis=0)
        Batch_Z_Count = tf.reduce_sum(self.Z_Onehot, axis=0)
        Z_Count_op = tf.assign_add(Z_Count, Batch_Z_Count)
        with tf.control_dependencies([Z_Count_op]):
            self.Z_Activation_op = tf.assign_add(Z_GateSum, Batch_Z_Gate)


class Mixture_of_experts:


    def __init__(self,X_img, Y_onehot, num_experts, num_classes, gatef, expertf,
                 Z_Onehot=None):
        self.X = X_img
        self.Y_Onehot = Y_onehot
        self.Y_True = tf.argmax(Y_onehot, axis=1, output_type=tf.int32)

        self.Z_Onehot = Z_Onehot
        if Z_Onehot is not None: self.Z_True = tf.argmax(Z_Onehot, axis=1, output_type=tf.int32)
        else: self.Z_True = None

        self.Gate = gatef(X=X_img, num_experts= num_experts, Z_Onehot=Z_Onehot)
        self.Largest_gate = tf.argmax(self.Gate, axis=1, output_type=tf.int32)

        if isinstance(expertf,list):
            self.Experts = [ef(X_img, num_classes) for ef in expertf]
        else:
            self.Experts = [expertf(X_img, num_classes) for _ in range(num_experts)]

        self.Y_Prob = sum([self.Gate[:,i:i+1]*Expert for i,Expert in enumerate(self.Experts)])
        self.Y_Pred = tf.argmax(self.Y_Prob, axis=1, output_type=tf.int32)
        self.Loss = -tf.reduce_sum(tf.math.xlogy(self.Y_Onehot,self.Y_Prob),axis=1)
        self.Train_op = tf.train.AdamOptimizer().minimize(self.Loss)

    def print(self):
        print("X",self.X)
        print("Y",self.Y_True)
        print("Z",self.Z_True)
        print("G",self.Gate)
        print("maxG", self.Largest_gate)
