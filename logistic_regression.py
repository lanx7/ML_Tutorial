import numpy as np
import random

def logisticRegression(X, y, theta, alpha, numIteration):
    m,n = np.shape(X)
    cost_history = np.zeros(numIteration)
    for i in range(numIteration):
        h = sigmoid(np.dot(X,theta))

        y1_term = -y * np.log(h)
        y0_term = (1-y) * np.log(1-h)
        cost = np.sum(y1_term-y0_term) / m

        #print np.shape(y1_term)
        #print np.shape(y0_term)
        #print np.shape(cost)


        cost_history[i] = cost
        print("Iteration %d | Cost: %f" % (i, cost))

        diff = h - y
        gradient = np.dot(X.T, diff) / m
        theta = theta - alpha * gradient
    return theta

def logisticRegressionReg(X, y, theta, alpha, numIteration,la):
    #theta_reg = theta.copy()
    #theta_reg[0] = 0


    m,n = np.shape(X)
    cost_history = np.zeros(numIteration)
    for i in range(numIteration):
        theta_reg = theta.copy()
        theta_reg[0] = 0
        h = sigmoid(np.dot(X,theta))

        y1_term = -y * np.log(h)
        y0_term = (1-y) * np.log(1-h)
        reg_term = la / (2 * m) * np.sum(theta_reg ** 2)
        #print "reg_term", reg_term
        cost = np.sum(y1_term-y0_term) / m + reg_term

        cost_history[i] = cost
        print("Iteration %d | Cost: %f" % (i, cost))

        reg_grad = (alpha * la / m) * theta_reg
        #print "reg_grad", reg_grad, l, m

        diff = h - y
        gradient = np.dot(X.T, diff) / m
        theta = theta - (alpha * gradient + reg_grad)
        #print theta
    return theta


def sigmoid(inX):
    return 1.0 / (1+ np.exp(-inX))

def predict(theta, X):
    m,n = np.shape(X)
    p = np.zeros(m)

    h = sigmoid(np.dot(X, theta))
    for i in range(np.size(h)):
        if h[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p


def sga(dataMatIn, classLabels):
    m,n = np.shape(dataMatIn)
    alpha = 0.1
    theta = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatIn[i]*theta))
        loss = classLabels[i] - h
        theta = theta + alpha * loss * dataMatIn[i]
    return theta

def sga1(dataMatIn, classLabels, numIter=150):
    m,n = np.shape(dataMatIn)
    theta = np.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatIn[randIndex]*theta))
            loss = classLabels[randIndex] - h
            theta = theta + alpha * loss * dataMatIn[randIndex]
            del(dataIndex[randIndex])
    return theta
