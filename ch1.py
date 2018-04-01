import numpy as np

a = np.random.rand(4,4)
print(a)

m = np.mat(a)
print(m) 

invm = m.I
print(invm)

identitym = m * invm
print(identitym)

identitym_ = np.eye(4)
print(identitym_)