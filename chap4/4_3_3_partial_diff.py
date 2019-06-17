# 편미분 : 분수가 여럿인 함수에 대한 미분


import numpy as np
import matplotlib.pylab as plt


def function_2(x):
    return x[0] ** 2 + x[1] ** 2
    # return np.sum(x ** 2)


def function_temp1(x0):
    return x0 * x0 + 4.0 ** 2.0


def function_temp2(x1):
    return 3.0 ** 2.0 + x1 * x1


def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


print(numerical_diff(function_temp1, 3.0))
