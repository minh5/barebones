import pdb

import numpy as np

from models import LinearGradientDescent, LogisticGradientDescent

def generate_normal_random_sets(size, n_sets):
    results = [np.ones(size)]
    for n in range(1, n_sets):
        results = np.append(results, [np.random.normal(size=size)], axis=0)
    return results

print('RUNNING LOGIT TEST')
x_set = [1, 2, 4, 3, 5]
y_set = [1, 3, 3, 2, 5]

test = LinearGradientDescent(x_set, y_set, 20)
test.run()

print('RUNNING LOGIT TEST')
n_size = 100
x_set = generate_normal_random_sets(n_size, 3)
y_set = np.random.binomial(1, .85, size=n_size)
test2 = LogisticGradientDescent(x_set, y_set, iterations=500)

try:
    test2.run()
except:
    pdb.post_mortem()
