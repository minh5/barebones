
class KNN:

    def __init__(self, x_train, y_train, x_test, neighbors=5):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.neighbors = neighbors

    @staticmethod
    def euclidean_distance(first, second):
        distance = 0
        for fi, sec in zip(first, second):
            distance += (fi - sec)**2
        return distance**(1/2)

    def get_k_neighbors(self):
        distances = []
        for row, y in zip(self.x_train, self.y_train):
            dist = self.euclidean_distance(self.x_test, row)
            distances.append((row, dist, y))
        distances.sort(key=lambda x: x[1])
        neighbors, scores = list(), list()
        for i in range(self.neighbors):
            neighbors.append(distances[i])
        return neighbors

    @staticmethod
    def make_prediction(neighbors, test):
        scores = []
        for i in neighbors:
            scores.append(i[2])
        return max(set(scores), key=scores.count)

    def run(self):
        neighbors = self.get_k_neighbors()
        return self.make_prediction(neighbors, self.x_test)
