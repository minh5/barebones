
class LVQ:

    def __init__(self, x_train, y_train, test):
        self.x_train = x_train
        self.y_train = y_train
        self.test = test

    @staticmethod
    def euclidean_distance(first, second):
        distance = 0
        for fi, sec in zip(first, second):
            distance += (fi - sec)**2
        return distance**(1/2)
