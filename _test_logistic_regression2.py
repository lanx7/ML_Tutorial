import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import *
import os
from ml_utils import *

PLOT_MODE = True

# Logistic regression test for data2 (Polynomial + Regularization)
data = np.genfromtxt('data/logistic_data2_reg.txt', delimiter=',')
X = data[:,[0,1]]
y = data[:,2]

if PLOT_MODE == True:
    pos = np.where(y==1)
    neg = np.where(y==0)
    plt.figure()
    plt.scatter(X[pos,0],X[pos,1], color='blue',marker='x', s=30,label="positive")
    plt.scatter(X[neg,0],X[neg,1], color='red',marker='o', s=30)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.draw()


X = mapFeature(X[:,0],X[:,1],6)
m,n = np.shape(X)
theta = np.zeros(n)


print X
print y

alpha = 0.03
numIteration = 3000000
l = 0.0 # lambda
theta = logisticRegressionReg(X,y,theta,alpha,numIteration,l)
print theta

predict = predict(theta,X)
print predict

accuracy = np.mean(predict == y) * 100
print "Accuracy is %f" % accuracy

if PLOT_MODE == True:
    u = np.linspace(-1,1.5,50)
    V = np.linspace(-1,1.5,50)
    z = np.zeros((np.size(u), np.size(V)))
    for i in range(np.size(u)):
        for j in range(np.size(V)):
            z[i,j] = np.dot(mapFeature(u[i],V[j]), theta)

    plt.contour(u,V,z,[0,0])
    plt.show()