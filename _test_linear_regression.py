import numpy as np
import matplotlib.pyplot as plt
from linear_regression import *
import os

PLOT_MODE = True

# Linear regresstion test for data1 (Single Feature)
data = np.genfromtxt('data/lr_data1.txt', delimiter=',')
X = data[:,0]
y = data[:,1]

m = np.size(X)
X = np.column_stack((np.ones(m),X))
theta = np.zeros(2)

alpha = 0.0001
numIteration = 1000
print "Training theta......."
theta, cost_history = gradientDescent(X, y, theta, alpha, numIteration)
print "Theta is converged to :", theta

if PLOT_MODE == True:
    # Scattering Data
    plt.scatter(X[:,1], y, marker="x")
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')

    # Regression Line
    plt.plot(X[:,1], np.dot(X,theta),'r')

    # Cost(J) changes as the learning goes
    plt.figure()
    plt.plot(range(0,numIteration), cost_history)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost(J)")
    plt.show()