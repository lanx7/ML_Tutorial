from data_utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from softmax_tf import *
import time
import tensorflow as tf

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
#y_onehot = onehot_transform(y)

num_classes = 10

alpha = 0.01
numIterations = 100
opt_theta = softmax(X, y, alpha, numIterations)


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


