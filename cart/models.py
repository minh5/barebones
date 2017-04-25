
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
        left, right, y_left, y_right = np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        for row, y in zip(x_set, y_set):
            if row[index] < value:
                left = np.append(left, row)
                y_left = np.append(y_left, y)
            else:
                right = np.append(right, row)
                y_right = np.append(y_right, y)
        return left, right, y_left, y_right

    def determine_best_split(self, x_set, y_set):
        col_index, best_gini, best_value, groups = None, 1, 0, None
        for column in range(x_set.shape[1]):
            for row in x_set:
                l, r, y_l, y_r = self.get_split_points(column, x_set, y_set, row[column])
                gini = sum([self.calculate_split_gini(y_l),
                           self.calculate_split_gini(y_r)])
                print('gini is:', gini)
                if gini < best_gini:
                    col_index, best_gini, best_value, x_groups, y_groups = column, gini, row[column], [l, r], [y_l, y_r]
        return {'col_index': col_index, 'value': best_value, 'x': x_groups, 'y': y_groups}

    @staticmethod
    def terminal_node_value(group):
        outcomes = set(group)
        return max(outcomes, key=list(outcomes).count)

    def create_nodes(self, x, y, depth):
        node = self.determine_best_split(x, y)
        left, right = node['y'][0], node['y'][1]
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
        if depth >= self.tree_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.determine_best_split(node['x'], node['y'])
            self.create_child_nodes(node['x'], node['y'], depth+1)
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.determine_best_split(node['x'], node['y'])
            self.create_child_nodes(node['x'], node['y'], depth+1)

    def create_tree(self):
        root = self.determine_best_split()
        root = self.create_child_nodes(root, 1)
        return root

    def run(self):
        pass
