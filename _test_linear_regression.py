import numpy as np
from linear_regression import *

data = np.genfromtxt('data/ex1data1.txt', delimiter=',')
X = data[:,0]
y = data[:,1]
print X, np.shape(X)
print y, np.shape(y)

m = np.size(X)
X = np.column_stack((np.ones(m),X))
theta = np.zeros(2)
print X
print computeCost(X, y, theta)


alpha = 0.005
numIteration = 1000
theta = gradientDescent(X, y, theta, alpha, numIteration)
