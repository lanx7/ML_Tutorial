import numpy as np
import tensorflow as tf
from ml_utils import *


def softmax(X, y, alpha, numIteration):
    n,m = np.shape(X)
    num_classes = 10

    y_onehot = onehot_transform(y)

    x_input = tf.placeholder(tf.float32, [None, n])
    y_input = tf.placeholder(tf.float32, [None, num_classes])

    W = tf.Variable(tf.random_normal([n,num_classes], stddev=0.01))
    b = tf.Variable(tf.zeros([num_classes]))

    py_x = tf.matmul(x_input, W)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, y_input)) # compute mean cross entropy (softmax is applied internally)

    # Another way to implement softmax
    # p = tf.nn.softmax(tf.matmul(x_input, W) + b)
    # cost = -tf.reduce_sum(y_input*tf.log(p)) # cross_entropy


    train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
      sess.run(init)
      for i in range(numIteration):
        #batch_xs, batch_ys = mnist.train.next_batch(100)
        for start, end in zip(range(0, len(X.T), 128), range(128, len(X.T), 128)):
          result = sess.run([train_step,cost] , feed_dict={x_input: X.T[start:end], y_input: y_onehot.T[start:end]})

        #result = sess.run([train_step,cost], feed_dict={x_input: X.transpose(), y_input: y_onehot.transpose()})
        print "%d, Cost: %f" % (i, result[1])
        opt_theta = W.eval().transpose()

    return opt_theta


def softmaxPredict(theta, X):
    n,m = np.shape(X)
    num_classes = 10

    theta = theta.reshape(num_classes, n)   # shape of theta is transposed....

    """ Compute the class probabilities for each example """
    hypothesis    = np.exp(np.dot(theta, X))
    p = hypothesis / np.sum(hypothesis, axis = 0)

    """ Give the predictions based on probability values """
    predictions = np.argmax(p, axis = 0)

    return predictions



""" Code From TensorFlow Example
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Train
tf.initialize_all_variables().run()
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
"""
