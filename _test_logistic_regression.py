import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import *
import os

PLOT_MODE = True

# Logistic regression test for data1
data = np.genfromtxt('data/logistic_data1.txt', delimiter=',')
X = data[:,[0,1]]
y = data[:,2]

m,n = np.shape(X)
X = np.column_stack((np.ones(m),X))
theta = np.zeros(n+1)


print X
print y

alpha = 0.003
numIteration = 1000000
theta = logisticRegression(X,y,theta,alpha,numIteration)
print theta

if PLOT_MODE == True:
    pos = np.where(y==1)
    neg = np.where(y==0)
    plt.figure()
    plt.scatter(X[pos,1],X[pos,2], color='blue',marker='x', s=30,label="positive")
    plt.scatter(X[neg,1],X[neg,2], color='red',marker='o', s=30)
    plt.xlabel("X1")
    plt.ylabel("X2")

    #u = np.arange(20, 100, 0.1)
    #print u
    u = np.linspace(20,100,50)
    #print u2
    v = (-theta[0] - theta[1] * u) / theta[2] # z = w0 + w1x1 + w2x2
    plt.plot(u,v)
    plt.show()
