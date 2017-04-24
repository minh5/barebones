
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
    def get_split_points(col_set, y_set, value):
        assert len(col_set) == len(y_set)
        # import pdb
        # pdb.set_trace()
        left, right = np.empty(0), np.empty(0)
        for x in range(len(col_set)):
            if col_set[x] < value:
                left = np.append(left, y_set[x])
            else:
                right = np.append(right, y_set[x])
        return left, right

    @staticmethod
    def calculate_split_gini(split_list):
        classes = set(split_list)
        gini_prep = []
        for item in classes:
            subset = [t for t in split_list if t == item]
            gini_prep.append(len(subset))
        return gini(*gini_prep)

    def determine_best_split(self, x, y):
        best_gini, best_value, groups = 1, 0, None
        for column in range(x.shape[1]):
            columns = x[0:, column]
            for row in columns:
                le, re = self.get_split_points(columns, self.y, row)
                gini = sum([self.calculate_split_gini(le),
                           self.calculate_split_gini(re)])
                print('gini is:', gini)
                if gini < best_gini:
                    col, best_gini, best_value, groups = column, gini, row, [le, re]
        return {'index': col, 'value': best_value, 'group': groups}

    @staticmethod
    def terminal_node_value(group):
        outcomes = set(group)
        return max(outcomes, key=list(outcomes).count)

    def create_child_nodes(self, depth):
        node = self.determine_best_split()
        left, right = node['group']
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
        if depth >= self.tree_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.determine_best_split(left)
            self.create_child_nodes(node['left'], depth+1)
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.determine_best_split(left)
            self.create_child_nodes(node['right'], depth+1)

    def create_tree(self):
        root = self.determine_best_split()
        root = self.create_child_nodes(root, 1)
        return root

    def run(self):
        pass
