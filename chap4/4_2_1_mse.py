import numpy as np


def mean_squarded_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)


y = [0.1, 0.05, 0.6, 0.0, 0.05,
     0.1, 0.0, 0.1, 0.0, 0.0]
x = [0.1, 0.05, 0.1, 0.0, 0.05,
     0.1, 0.0, 0.6, 0.0, 0.0]

t = [0, 0, 1, 0, 0,
     0, 0, 0, 0, 0]

print(mean_squarded_error(np.array(y), np.array(t)))
print(mean_squarded_error(np.array(x), np.array(t)))