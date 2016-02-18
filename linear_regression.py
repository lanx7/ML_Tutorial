import numpy as np

def computeCost(X, y, theta):
    hypothesis = np.dot(X, theta)                   # (m by n) * (n by 1) --> (m by 1)
    diff = hypothesis - y                           # (m by 1) - (m by 1)
    #cost = sum(np.square(diff)) / (2 *np.size(y))   # (m by 1)  --> Scala, np.square: element-wise ^2
    cost = np.sum(diff ** 2) / (2 *np.size(y))
    return cost
    pass


def gradientDescent(X, y, theta, alpha, numIterations):
    m = np.size(y)
    cost_history = np.zeros(numIterations)
    #print np.shape(x), np.shape(y), np.shape(theta)
    for i in range(0, numIterations):
        hypothesis = np.dot(X, theta)
        diff = hypothesis - y
        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(diff ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        # gradient = np.dot(xTrans, loss) / m
        gradient = np.dot(X.T, diff) / m
        # update
        theta = theta - alpha * gradient

    print ("Theta: %f", theta)
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

