from data_utils import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from softmax import *
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse
import scipy.optimize
import time


num_classes = 10
X = loadMNISTImages('data/mnist/train-images-idx3-ubyte')
y = loadMNISTLabels('data/mnist/train-labels-idx1-ubyte')
n,m = np.shape(X)  # Note that training examples are n-by-m matrix
#print np.shape(y), type(y)

#showImage(X[:,0],y[0])
rand = np.random.RandomState(int(time.time()))
theta = 0.005 * np.asarray(rand.normal(size = (num_classes*n, 1)))
#theta = np.zeros((n,num_classes))

f, g = softmaxCost(theta, X, y)
print 'f', f
print 'g', np.shape(g)

opt_solution  = scipy.optimize.minimize(softmaxCost, theta,
                                            args = (X, y,), method = 'L-BFGS-B',
                                            jac = True, options = {'maxiter': 100})
#y_onehot = full(sparse(y, 1:m,1,num_classes,m));

opt_theta = opt_solution.x

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
