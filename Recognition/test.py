import numpy as np
a = np.array([1,2,3]).reshape(-1,1)
b = np.array([4,5,6]).reshape(-1,1)
c = [a,b]
print(c[0].shape)
c=np.array(c).reshape(-1,c[0].shape[0]).T
print(c)
