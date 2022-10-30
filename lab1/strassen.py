import numpy as np


def split_into_quarts(matrix: np.array):
    shape = matrix.shape
    assert shape[0] == shape[1]
    segment_size = shape[0] // 2
    a11 = matrix[0:segment_size, 0:segment_size]
    a12 = matrix[segment_size:, 0:segment_size]
    a21 = matrix[0:segment_size, segment_size:]
    a22 = matrix[segment_size:, segment_size:]

    return a11, a12, a21, a22



def strassen(matrix1: np.array, matrix2: np.array):
    pass