import numpy as np

a = [
  [2, 2, 0, 2, 0],
  [0, 2, 0, 1, 2],
  [1, 0, 0, 0, 1],
  [0, 1, 2, 1, 2],
  [1, 0, 2, 0, 1]
]

b = [
  [0, 1, 1, 2, 1],
  [1, 2, 0, 1, 2],
  [1, 0, 2, 0, 0],
  [2, 0, 1, 1, 0],
  [2, 1, 0, 2, 2]
]

c = [
  [0, 1, 1, 1, 0],
  [0, 2, 0, 1, 1],
  [2, 1, 1, 0, 0],
  [0, 1, 0, 0, 0],
  [0, 0, 2, 1, 0]
]
x = np.array([a, b, c])

a = [
  [-1, 1, -1],
  [0, 1, 0],
  [-1, 0, 0]
]

b = [
  [-1, -1, 1],
  [1, 1, -1],
  [-1, 0, -1]
]

c = [
  [1, 1, 0],
  [0, -1, -1],
  [0, 0, 1]
]

w0 = np.array([a, b, c])
b0 = 1

a = [
  [-1, -1, -1],
  [1, 0, 1],
  [0, 0, 0]
]

b = [
  [1, -1, -1],
  [0, 0, 0],
  [-1, -1, -1]
]

c = [
  [-1, 1, -1],
  [0, -1, 0],
  [0, -1, 0]
]

w1 = np.array([a, b, c])
b1 = 0

stride = 2
pad = 1


w = np.array([w0, w1])
b = np.array([b0, b1])

npad = ((0, 0), (pad, pad), (pad, pad))
padded = np.pad(x, npad, 'constant', constant_values=(0))

C, H, W = x.shape
F, C, HH, WW = w.shape
new_H = 1 + int((H + 2 * pad - HH) / stride)
new_W = 1 + int((W + 2 * pad - WW) / stride)
tmp_padded = padded


out = np.zeros((F, new_H, new_W))
tmp = np.zeros((F, new_H, new_W))
for filter in xrange(F):
  for i in xrange(new_H * new_W):
    col = i % new_W
    row = int(i / new_W)
    root_col = col * stride
    root_row = row * stride
    v = 0
    local = tmp_padded[:, root_row: root_row + HH, root_col: root_col
                                                      + WW]
    v = np.sum(local * w[filter]) + b[filter]
    tmp[filter, row, col] = v
# print tmp
# print local.shape, w.shape
# print out
# print out.shape
# # print out.T.shape
# print 'local', local.shape, '\n'
# print 'w', w.shape, '\n'
# tmp = local[np.newaxis, :] * w
# print tmp.shape
# print np.sum(tmp, axis=1) + b

tmp = np.zeros((new_W, new_H, F))
for i in xrange(new_H * new_W):
  col = i % new_W
  row = int(i / new_W)
  root_col = col * stride
  root_row = row * stride
  local = tmp_padded[:, root_row: root_row + HH, root_col: root_col
                                                       + WW] * w
  tmp[col, row] = (np.sum(local, axis=(1, 2, 3)) + b).reshape((1, 2))
tmp = tmp.T
# print tmp

a = np.ones((3, 4))
c = np.array([1, 2, 0])
print (a.T * c).T