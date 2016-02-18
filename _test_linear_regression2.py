import numpy as np
import matplotlib.pyplot as plt
from linear_regression import *
from ml_utils import *

PLOT_MODE = True

# Linear regresstion test for data2 (Multiple Feature)
data = np.genfromtxt('data/lr_data2_multi.txt', delimiter=',')
X = data[:,[0,1]]
y = data[:,2]

X, mu, sigma = featureNormalize(X)
print X, mu, sigma

m,n = np.shape(X)
X = np.column_stack((np.ones(m),X))
theta = np.zeros(n+1)

alpha = 0.01
numIteration = 1000
print "Training theta......."
theta, cost_history = gradientDescent(X, y, theta, alpha, numIteration)
print "Theta is converged to :", theta

if PLOT_MODE == True:
    # Cost(J) changes as the learning goes
    plt.figure()
    plt.plot(range(0,numIteration), cost_history)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost(J)")
    plt.show()

# Data should be normalized before prediction
size = (1650 - mu[0]) / sigma[0]
bedroom = (3 - mu[1]) / sigma[1]
prediction = np.dot([1, size, bedroom], theta)
print "Predicted price of a 1650 sq-ft + 3 bedroom is %f" % prediction