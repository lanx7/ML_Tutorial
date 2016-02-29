from data_utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from softmax import *
import time
import tensorflow as tf

num_classes = 10
X = loadMNISTImages('data/mnist/train-images-idx3-ubyte')
y = loadMNISTLabels('data/mnist/train-labels-idx1-ubyte')
n,m = np.shape(X)  # Note that training examples are n-by-m matrix
print n,m
print np.shape(y), type(y)

#showImage(X[:,0],y[0])
rand = np.random.RandomState(int(time.time()))
theta = 0.005 * np.asarray(rand.normal(size = (num_classes*n, 1)))
#theta = np.zeros((n,num_classes))

num_classes = 10

y_onehot = onehot_transform(y)

theta = theta.reshape(num_classes, n)       # shape of theta is transposed....
print np.shape(y_onehot)
print np.shape(theta)

print X.transpose()[0:5,:]
print y_onehot.transpose()[0:5,:]

x_input = tf.placeholder(tf.float32, [None, 784])
y_input = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([784,10], stddev=0.01))
b = tf.Variable(tf.zeros([10]))

def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy

py_x = model(x_input, W)
#py_x = model(W, x_input)


#p = tf.nn.softmax(tf.matmul(x_input, W) + b)
#cost = -tf.reduce_sum(y_input*tf.log(p)) # cross_entropy

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, y_input)) # compute mean cross entropy (softmax is applied internally)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_input,py_x)) # compute mean cross entropy (softmax is applied internally)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.initialize_all_variables()
with tf.Session() as sess:
  sess.run(init)
  for i in range(100):
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    for start, end in zip(range(0, len(X.T), 128), range(128, len(X.T), 128)):
      result = sess.run([train_step,cost] , feed_dict={x_input: X.T[start:end], y_input: y_onehot.T[start:end]})

    #result = sess.run([train_step,cost], feed_dict={x_input: X.transpose(), y_input: y_onehot.transpose()})
    print "%d, Cost: %f" % (i, result[1])
    opt_theta = W.eval().transpose()

print opt_theta
print np.shape(opt_theta)

predictions = softmaxPredict(opt_theta, X)
correct = y == predictions
print "Training Accuracy: %f" % np.mean(correct)

""" Load MNIST test images and labels """

test_data   = loadMNISTImages('data/mnist/t10k-images-idx3-ubyte')
test_labels = loadMNISTLabels('data/mnist/t10k-labels-idx1-ubyte')

predictions = softmaxPredict(opt_theta, test_data)

correct = test_labels == predictions
print "Test Accuracy : %f" % np.mean(correct)
print predictions
print test_labels
