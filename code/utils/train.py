import tensorflow as tf
import numpy as np

def feed_iter(data_tuple, place_holder_tuple, datasize, batchsize=128, shuffle=True, mapping=lambda x:x):
    indices = np.random.permutation(datasize) if shuffle else np.arange(datasize)
    for b in range(0, datasize, batchsize):
        b_idx = indices[b:b+batchsize]
        b_datatuple = tuple(data[b_idx] for data in data_tuple)
        b_datatuple = mapping(b_datatuple)
        feeddict = dict(zip(place_holder_tuple, b_datatuple))
        yield feeddict


def train(sess, epochs, feed_iter_f, val_iter_f,
          trainstep,
          measure_op, measure,
          measure_reset_op=tf.local_variables_initializer):

    print("start train loop")

    train_measure_history = []
    valid_measure_history = []
    for e in range(epochs):
        print("epoch",e,"/",epochs)
        sess.run(measure_reset_op())
        print("train")
        for b, feed in enumerate(feed_iter_f()):
            if b % 128 == 127:
                print()
            print(end="-",flush=True)
            sess.run((trainstep, measure_op), feed)
        train_measure_history.append(sess.run(measure))
        print()
        print("measure validation")
        sess.run(measure_reset_op())
        for feed in val_iter_f():
            sess.run(measure_op, feed)
        valid_measure_history.append(sess.run(measure))
    return train_measure_history, valid_measure_history
