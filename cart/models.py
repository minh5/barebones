
import numpy as np
import pdb


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
        col_index, best_gini, best_value, groups = None, 1, 0, None
        for column in range(x_set.shape[1]):
            for row in x_set:
                print(row)
                l, r, y_l, y_r = self.get_split_points(column, x_set, y_set, row[column])
                gini = sum([self.calculate_split_gini(y_l),
                           self.calculate_split_gini(y_r)])
                print('gini is:', gini)
                if gini < best_gini:
                    x_groups, y_groups = dict(), dict()
                    x_groups['left'] = l
                    x_groups['right'] = r
                    y_groups['left'] = y_l
                    y_groups['right'] = y_r
                    col_index, best_gini, best_value = column, gini, row[column]
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

    def create_tree(self):
        root = self.determine_best_split(self.x, self.y)
        tree = self.create_nodes(root, 1)
        return root

    def run(self):
        pass
