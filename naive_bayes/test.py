import numpy as np

from models import NaiveBayes

x_train = np.array([
    [3.393533211, 2.331273381],
    [3.110073483, 1.781539638],
    [1.343808831, 3.368360954],
    [3.582294042, 4.67917911],
    [2.280362439, 2.866990263],
    [7.423436942, 4.696522875],
    [5.745051997, 3.533989803],
    [9.172168622, 2.511101045],
    [7.792783481, 3.424088941],
    [7.939820817, 0.791637231]
])

y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

test = NaiveBayes(x_train, y_train)
try:
    test.calculate_probs()
except BaseException:
    import pdb
    pdb.post_mortem()
