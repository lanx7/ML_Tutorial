__author__ = 'lanx'
import numpy as np
import random

def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T *xMat
    if np.linalg.det(xTx) == 0.0:
        print "This matrix is singular, connot do inverse"
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def gradientDescent(x, y, theta, alpha, m, numIterations):
    #print np.shape(x), np.shape(y), np.shape(theta)
    xTrans = x.transpose()
    #print np.shape(xTrans)
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
            #print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        # gradient = np.dot(xTrans, loss) / m
        gradient = np.dot(loss.transpose(), x) / m
        # update
        theta = theta - alpha * gradient
            #print ("Theta: %f", theta)
    return theta

def gradientDescentMatrix(x, y, theta, alpha, m, numIterations):
    xMat = np.mat(x)
    yMat = np.mat(y)
    tMat = np.mat(theta)
    for i in range(0, numIterations):
        # Gradient = X*XTw - Xy
        xsquare = np.dot(x.T, x)
        one = np.dot(xsquare,theta)
        two = np.dot(x.T, y)
        gradient = one - two
        theta = theta - alpha * gradient
            #print("Iteration %d | Theta: %f" % (i, theta))

    return theta

def gradRegressMatrix(xArr, yArr):
    xTemp = np.mat(xArr)
    yTemp = np.mat(yArr)
    xMat = np.squeeze(np.asarray(xTemp))
    yMat = np.squeeze(np.asarray(yTemp))
    #yMat = np.ones(m) * yArr #array scalar multiplication

    m, n = np.shape(xMat) #m = 100, n = 2

    #print yMat
    theta = np.ones(n)
    alpha = 0.0005
    numIteration = 100000
    #theta = gradientDescent(xMat, yMat, theta, alpha, m, numIteration)
    theta = gradientDescentMatrix(xMat, yMat, theta, alpha, m, numIteration)
    return theta

def gradRegres(xArr, yArr):
    xTemp = np.mat(xArr)
    yTemp = np.mat(yArr)
    xMat = np.squeeze(np.asarray(xTemp))
    yMat = np.squeeze(np.asarray(yTemp))
    #xMat = xTemp.getA()
    #yMat = yTemp.getA()
    #yMat = np.ones(m) * yArr #array scalar multiplication

    m, n = np.shape(xMat) #m = 100, n = 2

    #print yMat
    theta = np.ones(n)
    alpha = 0.0005
    numIteration = 100000
    theta = gradientDescent(xMat, yMat, theta, alpha, m, numIteration)
    #theta = gradientDescentMatrix(xMat, yMat, theta, alpha, m, numIteration)
    return theta



def logisticRegression(dataMatIn, classLabels):
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()

    m,n = np.shape(dataMat)
    alpha = 0.001
    numIteration = 500

    theta = np.ones((n,1))
    for k in range(numIteration):
        h = sigmoid(dataMat * theta)
        loss = (labelMat - h)

        gradient = dataMat.transpose() * loss
        theta = theta + alpha * gradient
    return theta

def sigmoid(inX):
    return 1.0 / (1+ np.exp(-inX))

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
