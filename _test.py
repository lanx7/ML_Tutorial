import numpy as np
import ml_utils
X = np.array([[1,2,3],[4,5,3],[7,8,9]])
index = [1,2]
print X[index]

index_col = [1,2]
print X[:,index]

index2 = np.where(X[:,2] == 3)
print X[:,index2[0]]

print X[1,2]
print X[1]

print range(1,6)

u = np.linspace(-1,1.5,50)
print u
print np.size(u)
print ml_utils.mapFeature(1,3)