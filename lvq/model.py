import random

import numpy as np

class LVQ:

    def __init__(
        self, x_train, y_train, test, n_codebooks=4,
        learning_rate=0.3, epochs=200
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.test = test
        self.n_codebooks = n_codebooks
        self.learning_rate = learning_rate
        self.epochs = epochs

    @staticmethod
    def euclidean_distance(first, second):
        distance = 0
        for fi, sec in zip(first, second):
            distance += (fi - sec)**2
        return distance**(1/2)

    def get_bmu(self, codebook, test_row):
        distances = list()
        for code in codebook:
            dist = self.euclidean_distance(code, test_row)
            distances.append((code, dist))
        distances.sort(key=lambda x: x[1])
        return distances[0]

    def initialize_codebook(self):
        codebook_x = []
        codebook_y = []
        for n in range(self.n_codebooks):
            codebook_row = []
            for i in range(0, self.x_train.shape[1]):
                rand_row = random.randrange(0, self.x_train.shape[0])
                codebook_row.append(self.x_train[rand_row][i])
            codebook_x.append(codebook_row)
            codebook_y.append(random.sample(self.y_train.tolist(), 1))
        return np.array(codebook_x), np.array(codebook_y)

    def linear_decay(self, epoch, total_epoch):
        return self.learning_rate * (1 - (epoch/total_epoch))

    def run(self):
        codebook_x, codebook_y = self.initialize_codebook()
        for epoch in range(self.epochs):
            sum_errors = 0
            rate = self.linear_decay(epoch, self.epochs)
            for row, y in zip(self.x_train, self.y_train):
                bmu = self.get_bmu(codebook_x, row)
                print('distance', bmu[1])
                bmu = bmu[0]
                index = codebook_x.tolist().index(bmu.tolist())
                bmu_y = codebook_y[index].item()
                for ro, bm in zip(range(len(row)), range(len(bmu))):
                    error = row[ro] - bmu[bm]
                    sum_errors += error**2
                    if y == bmu_y:
                        bmu[bm] += rate * error
                    else:
                        bmu[bm] -= rate * error
            print('learning rate:', rate)
            print('error', error)
