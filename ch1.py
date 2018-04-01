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

b = np.array([[1, 2], [3, 4]])
print(b)

bm = np.mat(b)
print(b)

b_4 = b * 4
print(b_4)
bm_4 = bm * 4
print(bm_4)

b_b = b * b
print(b_b)
bm_bm = bm * bm
print(bm_bm)

"""
The above gives two different results as when numpy arrays are multiplied, 
the multiplication is simply element-wise. But when numpy matrices are multiplied,
the multiplication is proper matrix multiplication.
"""