import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])

print(a.size)
print(b.size)
print(np.outer(a,b))
print(np.outer(a,b).shape)
print(np.mean(a))
