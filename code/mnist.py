from dataset import Dataset, onehot

from tensorflow.examples.tutorials.mnist import input_data
data_directory = "../data"
mnist = input_data.read_data_sets(data_directory)

mnist_train_y, mnist_test_y = map(onehot,(mnist.train.labels, mnist.test.labels))
mnist = Dataset(mnist.train.images, mnist_train_y, mnist.test.images, mnist_test_y)
#print("mnist:")
#mnist.print_shapes()
