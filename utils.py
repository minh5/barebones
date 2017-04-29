import pickle


def pickle_object(x, output_path):
    with open(output_path, 'wb') as output:
        pickle.dump(x, output, pickle.HIGHEST_PROTOCOL)


def read_pickle(filepath):
    with open('large_y.pickle', 'rb') as input:
        return pickle.load(input)
