import numpy as np

Z = np.array([[1, 2], [3, 4], [5, 6]])

print(Z)
print(np.ndim(Z))
print(Z.shape)

# Multiple Matrix
A = np.array([[1, 2], [3, 4]])
print(A.shape)

B = np.array([[5, 6], [7, 8]])
print(B.shape)
print(np.dot(A, B))

# Multiple Matrix - 2x3 * 3x2
C = np.array([[1, 2, 3], [4, 5, 6]])
print(C.shape)

D = np.array([[1, 2], [3, 4], [5, 6]])
print(D.shape)

print(np.dot(C, D))

E = np.array([[1, 2], [3, 4], [5, 6]])
print(E.shape)
F = np.array([7, 8])
print(F.shape)
print(np.dot(E, F))
