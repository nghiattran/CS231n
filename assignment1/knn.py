import numpy as np

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
b = np.array([[1,2,3,4], [5,6,7,8]])

# col
print a[:, 0]

# row
print b[1]

print b
# Transpose
print b.T

print b[1][:, np.newaxis]
print b[1][:, np.newaxis].shape

X_mask = np.zeros(b.shape)
X_mask[b % 2 == 0] = 1

print X_mask