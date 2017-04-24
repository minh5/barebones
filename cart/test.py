import pdb

import numpy as np
from models import Cart

x_set = np.array([[2.771244718, 1.784783929],
                 [1.728571309, 1.169761413],
                 [3.678319846, 2.81281357],
                 [3.961043357, 2.61995032],
                 [2.999208922, 2.209014212],
                 [7.497545867, 3.162953546],
                 [9.00220326, 3.339047188],
                 [7.444542326, 0.476683375],
                 [10.12493903, 3.234550982],
                 [6.642287351, 3.319983761]])

y_set = np.append(np.zeros(5), np.ones(5))
test = Cart(x_set, y_set)

try:
    test.determine_best_split(test.x, test.y)
except:
    pdb.post_mortem()

# test.determine_best_split(test.x, test.y)
