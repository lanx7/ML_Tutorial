import numpy as np
import matplotlib.pyplot as plt

def featureNormalize(X):
    X_norm = X.copy()
    m,n = np.shape(X)
    mu = np.zeros(n)
    sigma = np.zeros(n)

    for i in range(n):
        mu[i] = np.mean(X[:,i])
        sigma[i] = np.std(X[:,i])

        X_norm[:,i] = (X[:,i] - mu[i]) / sigma[i]

    return X_norm, mu, sigma


def mapFeature(x1,x2, degree=6):
    out = np.ones(np.size(x1))
    new = np.zeros(np.size(x1))
    for i in range(1,degree+1):
        for j in range(0,i+1):
            #print i-j, j
            new = (x1 ** (i-j)) * (x2 ** j)
            out = np.column_stack((out,new))
    return out

def plotDecisionBoundary(theta, X, y):
    u = np.linspace(-1,1.5,50)
    V = np.linspace(-1,1.5,50)
    z = np.zeros((np.size(u), np.size(V)))
    for i in range(np.size(u)):
        for j in range(np.size(V)):
            z[i,j] = np.dot(mapFeature(u[i],V[j]), theta)

    plt.contour(u,V,z,[0,0])
    plt.show()
