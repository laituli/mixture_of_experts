"""
file structure

root
|
|---models
| |
| |---D<dataset>
|   |
|   |---E<expert>
|     |
|     |---G<gate>
|       |
|       |---N<number_of_expert>
|         |
|         |---<timestamp>
|           |
|           |---epoch<epoch>.model
|           |---train_acc.txt
|           |---test_acc.txt
|
|---results
  |
  |---Experiment123
    |
    |---EvalTimestamp456
      |
      |---combined_resultABC

"""
import os
print(os.path.isdir(""))

"""
import tensorflow as tf

a = tf.constant(1)
with tf.Session() as sess:
    print(sess.run([]))
"""
