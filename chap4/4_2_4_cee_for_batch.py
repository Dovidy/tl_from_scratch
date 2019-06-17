import numpy as np


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    # return -np.sum(t * np.log(y + 1e-7)) / batch_size # one-hot-encoding
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size #num label


'''
y[np.arange(batch_size), t] : 각 데이터의 정답 레이블에 해당하는 신경망의 출력 추출 
'''

y = [0.1, 0.05, 0.6, 0.0, 0.05,
     0.1, 0.0, 0.1, 0.0, 0.0]
x = [0.1, 0.05, 0.1, 0.0, 0.05,
     0.1, 0.0, 0.6, 0.0, 0.0]

t = [0, 0, 1, 0, 0,
     0, 0, 0, 0, 0]


np.arange(5) # [0,1,2,3,4]
