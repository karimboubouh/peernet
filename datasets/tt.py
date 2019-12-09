import numpy as np

x = np.array([1,2,3,4])
x = x.reshape(x.shape[0], 1)
x = x.T
print(x.shape)
print(x)


y = np.array([5,6,7,8])
y = y.reshape(y.shape[0], 1)
y = y.T
print(y.shape)
print(y)


z = np.array()
z.append(x)
print(z)
print(z.shape)