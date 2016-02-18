import numpy as np
import matplotlib.pyplot as plt
from linear_regression import *
import os

PLOT_MODE = True

# Linear regression test for data3 (Single Feature)
# Delimiter is not ',' but'\t'
data = np.genfromtxt('data/lr_data3.txt', delimiter='\t')
X = data[:,[0,1]]
y = data[:,2]

m = np.size(X)

# Since x0 exist in data file. need not to add x0 column
# X = np.column_stack((np.ones(m),X))
theta = np.zeros(2)

alpha = 0.01
numIteration = 10000
print "Training theta......."
theta, cost_history = gradientDescent(X, y, theta, alpha, numIteration)
print "Theta is converged to :", theta

if PLOT_MODE == True:
    # Scattering Data
    plt.scatter(X[:,1], y, marker="x")

    # Regression Line
    plt.plot(X[:,1], np.dot(X,theta),'r')

    # Cost(J) changes as the learning goes
    plt.figure()
    plt.plot(range(0,numIteration), cost_history)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost(J)")
    plt.show()

