import numpy as np
import random

from base import covariance


def matrix_multiply(matrix1, matrix2):
    result = np.zeroes((matrix1.shape[0], matrix2.shape[1]))
    for i in range(matrix1.shape[0]):
        for j in range(matrix2.shape[1]):
            for k in range(matrix2.shape[0]):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result


def covariance_matrix(matrix):
    tranposed = matrix.transpose()
    results = np.zeros((matrix.shape[0], tranposed.shape[1]))
    for i in range(matrix.shape[0]):
        for j in range(tranposed.shape[1]):
            results[i][j] += covariance(matrix[i, :], tranposed[:, j])
    return results


def scale(col):
    return np.array([(i - np.mean(col)) / np.std(col) for i in col])


def rescale_matrix(matrix):
    rescaled = matrix.copy()
    for col in range(matrix.shape[1]):
        if np.std(matrix[:, col]) > 0:
            rescaled[:, col] = scale(matrix[:, col]).astype(float)
    return rescaled


def magnitude(vector):
    return (sum([v * v for v in vector]))**(1/2)


def direction(vector):
    mag = magnitude(vector)
    return [v / mag for v in vector]


def directional_variance(matrix, direction_vector):
    d = direction(direction_vector)
    variances = []
    for row in matrix:
        variances.append(sum([v * d_i for v, d_i in zip(row, d)]) ** 2)
    return sum(variances)


def scalar_multiply(scalar, vector):
    return [scalar * v for v in vector]


def vector_distance(vector1, vector2):
    return sum([(v_1 - v_2)**2 for v_1, v_2 in zip(vector1, vector2)]) ** (1/2)


def find_eigenvector(matrix, tolerance=0.00001):
    guess = [random.random() for __ in matrix]
    while True:
        result = matrix_multiply(matrix, guess)
        length = magnitude(result)
        next_guess = scalar_multiply(1/length, result)
        if vector_distance(guess, next_guess) < tolerance:
            return next_guess, length  # eigenvector, eigenvalue
        guess = next_guess


def pca(matrix, n_components):
    rescaled = rescale_matrix(matrix)
    covar = covariance_matrix(rescaled)
    e_vector, e_value = find_eigenvector(covar)
    e_value_sorted = sorted(e_value.tolist())
    indices = []
    for e in e_value_sorted:
        indices.append(e_value.tolist().index(e), reverse=True)
    pdp = e_vector[: indices]
    component_matrix = matrix_multiply(rescaled, pdp)
    # need to add some stuff on selection n_components from resulting matrix
