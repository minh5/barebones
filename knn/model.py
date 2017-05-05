
class KNN:

    def __init__(self, x_train, y_train, x_test, y_st, neighbors=5):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.neighbors = neighbors

    @staticmethod
    def euclidean_distance(first, second):
        # assert len(first) == len(second)
        distance = 0
        for a, b in zip(first, second):
            distance += (a - b)**2
        return distance**(1/2)

    def get_neighbors(self):
        distances = []
        for row in self.x_train:
            dist = self.euclidean_distance(self.x_test, row)
            distances.append((row, dist))
        distances.sort(key=lambda x: x[1])
        neighbors = []
        for i in range(self.neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    def run(self):
        pass
