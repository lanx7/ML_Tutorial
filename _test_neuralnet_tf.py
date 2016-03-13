from data_utils import *
from ml_utils import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from neuralnet_tf import *
import time

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

num_classes = 10
X = loadMNISTImages('data/mnist/train-images-idx3-ubyte')
y = loadMNISTLabels('data/mnist/train-labels-idx1-ubyte')
n,m = np.shape(X)  # Note that training examples are n-by-m matrix
print n,m
print np.shape(y), type(y)

#showImage(X[:,0],y[0])
#rand = np.random.RandomState(int(time.time()))
#theta = 0.005 * np.asarray(rand.normal(size = (num_classes*n, 1)))
#theta = np.zeros((n,num_classes))
y_onehot = onehot_transform(y)

num_classes = 10
hidden_num = 625

alpha = 0.05
numIterations = 100
#opt_theta = softmax(X, y, alpha, numIterations)

x_input = tf.placeholder(tf.float32, [None, n])
y_input = tf.placeholder(tf.float32, [None, num_classes])

w_h = init_weights([784, 625]) # create symbolic variables
w_o = init_weights([625, 10])

py_x = model(x_input, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, y_input)) # compute costs
train_op = tf.train.GradientDescentOptimizer(alpha).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(numIterations):
    for start, end in zip(range(0, len(X.T), 128), range(128, len(X.T), 128)):
        result = sess.run([train_op,cost], feed_dict={x_input: X.T[start:end], y_input: y_onehot.T[start:end]})

    print "%d Iteration: Cost %f" % (i, result[1])
    #print np.argmax(y_onehot.T, axis=1)
    #print sess.run(predict_op, feed_dict={x_input: X.T, y_input: y_onehot.T})
    print np.mean(np.argmax(y_onehot.T, axis=1) ==
                     sess.run(predict_op, feed_dict={x_input: X.T, y_input: y_onehot.T}))
