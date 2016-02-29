
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression_tf import *
import os
from ml_utils import *

PLOT_MODE = False

# Logistic regression test for data2 (Polynomial + Regularization)
data = np.genfromtxt('data/logistic_data1.txt', delimiter=',')
#data = np.genfromtxt('data/logistic_data3.txt', delimiter='\t')

X = data[:,[0,1]]
y = data[:,2]

X, mu, sigma = featureNormalize(X)
print X, mu, sigma


m,n = np.shape(X)
X = np.column_stack((np.ones(m),X))
theta = np.zeros(n+1)

Y = np.zeros([m,1])
Y[:,0] = data[:,2]

print X
print y

print m,n
alpha = 0.003
numIterations = 100000

if PLOT_MODE == True:
    pos = np.where(Y[:,0]==1)
    neg = np.where(Y[:,0]==0)
    plt.figure()
    plt.scatter(X[pos,1],X[pos,2], color='blue',marker='x', s=30,label="positive")
    plt.scatter(X[neg,1],X[neg,2], color='red',marker='o', s=30)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

theta = logisticRegression(X, Y, theta, alpha, numIterations)

predict = predict(theta,X)
print predict

accuracy = np.mean(predict == Y[:,0]) * 100
print "Accuracy is %f" % accuracy