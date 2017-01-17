import numpy as np

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
b = np.array([[1,2,3,4], [5,6,7,8]])
c = np.array([
  [0, 3],
  [1, 2],
  [5, 6]
])
d = np.array([1, 0, 1])
x = np.array([3, 10, 2, 8, 9, 7, 6, 1])
y = np.array([
  [1, 1],
  [3, 4],
  [3, 3]
])

# get 3, 1, 6
print d[np.arange(3)]
correct_labels = c[np.arange(3), d[np.arange(3)]].reshape((3,1))
print correct_labels
print c - correct_labels

print np.count_nonzero(b, axis=0)
print np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]], axis=1)