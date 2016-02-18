import numpy as np

def gradientDescent(X, y, theta, alpha, numIterations):
    m = np.size(y)
    cost_history = np.zeros(numIterations)
    for i in range(0, numIterations):
        hypothesis = np.dot(X, theta)            # (m by n) * (n by 1) --> (m by 1)
        diff = hypothesis - y                    # (m by 1) - (m by 1)

        # avg cost per example, cost itself is not used for updating theta.
        # cost_history is useful for debugging the learning rate
        cost = np.sum(diff ** 2) / (2 * m)      # m by 1 -> Scala by vector summation
        cost_history[i] = cost
        print("Iteration %d | Cost: %f" % (i, cost))

        # avg gradient per example, (n by m) * (m by 1) --> n by 1 vector (i.e., feature-wise vector)
        gradient = np.dot(X.T, diff) / m

        # update theta
        theta = theta - alpha * gradient

    #print ("Theta: %f", theta)
    return theta, cost_history
