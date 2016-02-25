import numpy as np
import matplotlib.pyplot as plt
from linear_regression_tf import *
import os

PLOT_MODE = True

# Linear regresstion test for data1 (Single Feature)
data = np.genfromtxt('data/lr_data1.txt', delimiter=',')
X = data[:,0]
m = np.size(X)
X = np.column_stack((np.ones(m),X))

Y = np.zeros((m,1))
Y[:,0] = data[:,1]
theta = np.zeros((2,1))

alpha = 0.01
numIteration = 10000
print "Training theta......."
theta = linearRegression(X,Y,theta,alpha,numIteration)

#theta, cost_history = gradientDescent(X, y, theta, alpha, numIteration)
print "Theta is converged to :", theta


data_in1 = [1, 3.5]
prediction1 = np.dot(data_in1, theta)
print "Predicted price at 35000 people is %f" % (prediction1 * 10000)

data_in2 = [1, 7]
prediction2 = np.dot(data_in2, theta)
print "Predicted price at 70000 people is %f" % (prediction2 * 10000)

if PLOT_MODE == True:
    # Scattering Data
    plt.scatter(X[:,1], Y[:,0], marker="x")
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')

    # Regression Line
    plt.plot(X[:,1], np.dot(X,theta),'r')

    # Prediction point
    plt.plot(3.5, prediction1, marker='o', color='red')
    plt.plot(7, prediction2, marker='o', color='red')
    plt.show()

"""
    # Cost(J) changes as the learning goes
    plt.figure()
    plt.plot(range(0,numIteration), cost_history)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost(J)")
    plt.show()
"""
