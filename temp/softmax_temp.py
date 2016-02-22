# This piece of software is bound by The MIT License (MIT)
# Copyright (c) 2014 Siddharth Agrawal
# Code written by : Siddharth Agrawal
# Email ID : siddharth.950@gmail.com

import struct
import numpy
import array
import time
import scipy.sparse
import scipy.optimize

from data_utils import *
from ml_utils import *

###########################################################################################
""" The Softmax Regression class """

class SoftmaxRegression(object):

    #######################################################################################
    """ Initialization of Regressor object """

    def __init__(self, input_size, num_classes, lamda):

        """ Initialize parameters of the Regressor object """

        self.input_size  = input_size  # input vector size
        self.num_classes = num_classes # number of classes
        self.lamda       = lamda       # weight decay parameter

        """ Randomly initialize the class weights """

        rand = numpy.random.RandomState(int(time.time()))

        self.theta = 0.005 * numpy.asarray(rand.normal(size = (num_classes*input_size, 1)))

    #######################################################################################
    """ Returns the cost and gradient of 'theta' at a particular 'theta' """

    def softmaxCost(self, theta, X, y):

        """ Compute the groundtruth matrix """

        #ground_truth = self.getGroundTruth(labels)
        y_onehot = onehot_transform(y)

        """ Reshape 'theta' for ease of computation """

        theta = theta.reshape(self.num_classes, self.input_size)
        #theta = theta.reshape(self.input_size, self.num_classes)

        theta_x       = numpy.dot(theta, X)
        #theta_x       = numpy.dot(theta.T, X)
        hypothesis  = np.exp(theta_x)
        p = hypothesis / np.sum(hypothesis,axis=0) # colsum...


        """ Compute the class probabilities for each example """


        """ Compute the traditional cost term """
        #cost_examples    = numpy.multiply(y_onehot, numpy.log(p))
        #traditional_cost = -(numpy.sum(cost_examples) / X.shape[1])
        #traditional_cost = -sum(sum(np.multiply(y_onehot,np.log(p))))

        cost = -sum(sum(np.multiply(y_onehot,np.log(p))))

        """ Compute the weight decay term """

        #theta_squared = numpy.multiply(theta, theta)
        #weight_decay  = 0.5 * self.lamda * numpy.sum(theta_squared)

        """ Add both terms to get the cost """

        #cost = traditional_cost + weight_decay

        """ Compute and unroll 'theta' gradient """
        #gradient = np.dot(-X,(y_onehot - hypothesis).T)


        gradient = -numpy.dot(y_onehot - p, X.T)
        #gradient = gradient / X.shape[1] + self.lamda * theta
        gradient = numpy.array(gradient)
        gradient = gradient.flatten()

        return [cost, gradient]

    #######################################################################################
    """ Returns predicted classes for a set of inputs """

    def softmaxPredict(self, theta, input):

        """ Reshape 'theta' for ease of computation """

        theta = theta.reshape(self.num_classes, self.input_size)

        """ Compute the class probabilities for each example """

        theta_x       = numpy.dot(theta, input)
        hypothesis    = numpy.exp(theta_x)
        probabilities = hypothesis / numpy.sum(hypothesis, axis = 0)

        """ Give the predictions based on probability values """

        #predictions = numpy.zeros((input.shape[1], 1))
        #predictions[:, 0] = numpy.argmax(probabilities, axis = 0)
        predictions = numpy.argmax(probabilities, axis = 0)

        return predictions

###########################################################################################
""" Loads the images from the provided file name """


def executeSoftmaxRegression():

    """ Initialize parameters of the Regressor """

    input_size     = 784    # input vector size
    num_classes    = 10     # number of classes
    lamda          = 0.0001 # weight decay parameter
    max_iterations = 100    # number of optimization iterations

    """ Load MNIST training images and labels """

    training_data   = loadMNISTImages('data/mnist/train-images-idx3-ubyte')
    training_labels = loadMNISTLabels('data/mnist/train-labels-idx1-ubyte')

    """ Initialize Softmax Regressor with the above parameters """

    regressor = SoftmaxRegression(input_size, num_classes, lamda)

    """ Run the L-BFGS algorithm to get the optimal parameter values """

    opt_solution  = scipy.optimize.minimize(regressor.softmaxCost, regressor.theta,
                                            args = (training_data, training_labels,), method = 'L-BFGS-B',
                                            jac = True, options = {'maxiter': max_iterations})
    opt_theta     = opt_solution.x

    """ Load MNIST test images and labels """

    test_data   = loadMNISTImages('data/mnist/t10k-images-idx3-ubyte')
    test_labels = loadMNISTLabels('data/mnist/t10k-labels-idx1-ubyte')

    """ Obtain predictions from the trained model """

    predictions = regressor.softmaxPredict(opt_theta, test_data)

    """ Print accuracy of the trained model """

    correct = test_labels == predictions
    print test_labels
    print predictions
    print """Accuracy :""", numpy.mean(correct)

executeSoftmaxRegression()