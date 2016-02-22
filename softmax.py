import numpy as np
from ml_utils import *

def softmaxCost(theta, X, y):
    n,m = np.shape(X)
    num_classes = 10

    y_onehot = onehot_transform(y)

    theta = theta.reshape(num_classes, n)       # shape of theta is transposed....
    hypothesis  = np.exp(np.dot(theta, X))
    p = hypothesis / np.sum(hypothesis,axis=0) # colsum...

    #print 'y_onehot', np.shape(y_onehot), type(y_onehot)
    #print 'theta', np.shape(theta), type(theta)        # 10 * 784
    #print 'h', np.shape(hypothesis),type(hypothesis)   # 10 * 60000 type(hypothesis)
    #print 'p', np.shape(p), type(p)

    cost = -sum(sum(np.multiply(y_onehot,np.log(p))))

    gradient = -np.dot(y_onehot - p, X.T)
    #gradient = -np.dot(X,(y_onehot - hypothesis).T)

    # print np.shape(gradient)                            # 10 * 784

    gradient = gradient.flatten()
    return cost, gradient

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


