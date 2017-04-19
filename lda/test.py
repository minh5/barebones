import numpy as np

high = np.random.normal(10, size=100)
low = np.random.normal(1, size=100)
x_set = np.append(high, low)

ones = np.ones(100)
zeros = np.zeros(100)
y_set = np.append(ones, zeros)
