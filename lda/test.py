import pdb
import numpy as np

from models import LDA

high_mean = 10
low_mean = 1
high = np.random.normal(high_mean, size=100)
low = np.random.normal(low_mean, size=100)
x_train_set = np.append(high, low)

ones = np.ones(100)
zeros = np.zeros(100)
y_train_set = np.append(ones, zeros)


x_test_set = np.random.normal(high_mean, 10)
y_test_set = np.random.normal(low_mean, 10)
x_test_set = np.append(x_test_set, y_test_set)
ones = np.ones(10)
zeros = np.zeros(10)
y_test_set = np.append(ones, zeros)

test = LDA(x_train_set, y_train_set, x_test_set, y_test_set)

try:
    test.run()
except:
    pdb.post_mortem()
