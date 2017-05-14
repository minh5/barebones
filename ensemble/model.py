from math import log, exp

import numpy as np


def gini(*args):
    results = []
    for arg in args:
        results.append(arg/sum(args))
    return sum([k * (1-k) for k in results])


class Cart:

    def __init__(self, x, y, tree_depth=2, min_nodes=2):
        self.x = x
        self.y = y
        self.tree_depth = tree_depth
        self.min_nodes = min_nodes
        self.tree = self.run()

    @staticmethod
    def calculate_split_gini(split_list):
        classes = set(split_list)
        gini_prep = []
        for item in classes:
            subset = [t for t in split_list if t == item]
            gini_prep.append(len(subset))
        return gini(*gini_prep)

    @staticmethod
    def get_split_points(index, x_set, y_set, value):
        left, right, y_left, y_right = np.zeros(x_set.shape[1]), np.zeros(x_set.shape[1]), list(), list()
        for row, y in zip(x_set, y_set):
            if row[index] < value:
                left = np.vstack((left, row))
                y_left.append(y)
            else:
                right = np.vstack((right, row))
                y_right.append(y)
        return left[1:], right[1:], y_left, y_right

    def determine_best_split(self, x_set, y_set):
        col_index, best_gini, best_value, x_groups, y_groups = None, 1, 0, dict(), dict()
        for column in range(x_set.shape[1]):
            for row in x_set:
                l, r, y_l, y_r = self.get_split_points(column, x_set, y_set, row[column])
                gini = sum([self.calculate_split_gini(y_l),
                           self.calculate_split_gini(y_r)])
                if gini < best_gini:
                    x_groups['left'] = l
                    x_groups['right'] = r
                    y_groups['left'] = y_l
                    y_groups['right'] = y_r
                    col_index, best_gini, best_value = column, gini, row[column]
                    print('best gini', best_gini)
        return {'col_index': col_index, 'value': best_value, 'x': x_groups, 'y': y_groups}

    @staticmethod
    def terminal_node_value(group):
        outcomes = set(group)
        return max(outcomes, key=list(outcomes).count)

    def create_nodes(self, node, depth):
        left, right = node['y']['left'], node['y']['right']
        if not left or not right:
            node['left'] = node['right'] = self.terminal_node_value(left + right)
        if depth >= self.tree_depth:
            node['left'], node['right'] = self.terminal_node_value(left), self.terminal_node_value(right)
            return
        if len(left) <= self.min_nodes:
            node['left'] = self.terminal_node_value(left)
        else:
            node['left'] = self.determine_best_split(node['x']['left'], node['y']['left'])
            self.create_nodes(node['left'], depth+1)
        if len(right) <= self.min_nodes:
            node['right'] = self.terminal_node_value(right)
        else:
            node['right'] = self.determine_best_split(node['x']['right'], node['y']['right'])
            self.create_nodes(node['right'], depth+1)
        return node

    def predict(self, node, input_value):
        if input_value[node['col_index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], input_value)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], input_value)
            else:
                return node['right']

    def create_tree(self):
        root = self.determine_best_split(self.x, self.y)
        tree = self.create_nodes(root, 1)
        return tree


class BaggedCart(Cart):

    def __init__(self, sample_ratio, n_trees, bootstrap=True):
        self.sample_ratio = sample_ratio
        self.n_trees = n_trees
        self.bootstrap = boostrap

    def sample_x(self):
        indx = np.random.choice(
            self.x.shape[0],
            size=self.x_train.shape[0] * self.sample_ratio,
            replace=self.boostrap
        )
        self.x = self.x[indx]

    def run(self):
        self.trees = []
        for n in range(self.n_trees):
            train_set = self.sample_x()
            tree = self.create_tree()
            self.trees.append(tree)

    def predict_bagged_trees(self, input_value):
        scores = []
        for tree in self.trees:
            scores.append(self.predict(tree, input_value))
        return max(set(scores), key=scores.count)


class RandomForest(Cart):

    def __init__(self, n_features='auto', bootstrap, sample_ratio):
        self.n_features = n_features
        self.boostrap = boostrap
        self.sample_ratio = self.sample_ratio

    def sample_x(self):
        if self.n_features == 'auto':
            n_features = round(self.x_train.shape[1]**(1/2))
        else:
            n_features = self.n_features
        indx = np.random.choice(
            self.x.shape[0],
            size=self.x_train.shape[0] * self.sample_ratio,
            replace=self.boostrap
        )
        row_sampled = self.x[indx]
        sampled_df = []
        for row in row_sampled:
            sampled_df.append(random.sample(row.tolist(), n_features))
        self.x = np.array(sample_df)

    def run(self):
        self.trees = []
        for n in range(self.n_trees):
            train_set = self.sample_x()
            tree = self.create_tree()
            self.trees.append(tree)

    def predict_random_forest(self, input_value):
        scores = []
        for tree in self.trees:
            scores.append(self.predict(tree, input_value))
        return max(set(scores), key=scores.count)

    def AdaBoost(Cart):

        def __init__(self):
            self.tree_depth = 1


class AdaBoost(BaggedCart):

    def __init__(self):
        self.tree_depth = 1
        self.n_learners = 3
        self.weights = self.initialize_weights()
        self.stage = 1

    def initialize_weights(self):
        return [1/len(self.y) for i in len(self.y)]

    @staticmethod
    def calculate_stage(error):
        return log((1-error)/error)

    @staticmethod
    def update_weights(weight, stage):
        return weight * exp(stage * weight)

    def create_weak_learners(self):
        self.run()  # create weak learners through shallow trees

    def update_stages(self, tree, weights, stage_value):
        predictions, weighted_errors = [], []
        for row in self.x:
            predictions.append(self.predict(tree, row))
        for i in range(len(predictions)):
            if predictions[i] == self.y[i]:
                weighted_errors.append(0)
            else:
                new_weights = self.update_weights(weights[i], stage_value)
                weighted_errors.append(new_weights)
        self.stage = self.calculate_stage(sum(weighted_errors)/len(self.y))
        self.weights = new_weights

    @staticmethod
    def _predict_adaboost(tree, row, stage):
        predictions = self.predict(tree, row)
        if prediction == 1:
            return stage * 1
        else:
            return stage * -1

    def predict_adaboost(self, row):
        ensembles = []
        for tree in self.trees:
            predict = self._predict_adaboost(tree, row, self.stage)
            ensembles.append(predict)
        if sum(ensembles) > 0:
            return 1
        else:
            return 0

    def run(self, predict_row=None):
        all_trees = []
        self.create_weak_learners()
        for tree in self.trees:
            self.update_stages(tree, self.weights, self.stage)
