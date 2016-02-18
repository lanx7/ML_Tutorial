import numpy as np

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


