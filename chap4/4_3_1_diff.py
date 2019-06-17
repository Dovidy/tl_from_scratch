

# bad realization ex
def bad_numerical_diff(f, x):
    h = 10e-50 # 소수점 아래 0이 50개
    return (f(x + h) - f(x)) / h

# 중심차분, 반올림 오차 해결
def good_numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)