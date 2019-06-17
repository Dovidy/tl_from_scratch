import numpy as np


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) # delta는 np.log(0)=-inf 방지하기 위해  필요


y = [0.1, 0.05, 0.6, 0.0, 0.05,
     0.1, 0.0, 0.1, 0.0, 0.0]
x = [0.1, 0.05, 0.1, 0.0, 0.05,
     0.1, 0.0, 0.6, 0.0, 0.0]

t = [0, 0, 1, 0, 0,
     0, 0, 0, 0, 0]


print(cross_entropy_error(np.array(y), np.array(t)))
print(cross_entropy_error(np.array(x), np.array(t)))