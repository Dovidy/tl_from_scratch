import numpy as np


def softmax(a):
    c = np.max(a) # overflow solution
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


a = np.array([0.3, 2.9, 4.0])
print(softmax(a))
print(np.sum(softmax(a)))

# 3.5.2 softmax overflow

b = np.array([1010, 1000, 990])
# print(softmax(b)) OVERFLOW
b = b - np.max(b)
print(softmax(b))
print(np.sum(softmax(b)))

# 3.5.3 feature of softmax
# sum of softmax's output is 1

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))


