import numpy as np

matrix_t = np.ndarray


float_op_cnt = {'*': 0, '/': 0, '+': 0, '-': 0}


def split_into_quarts(matrix: matrix_t) \
        -> tuple[matrix_t, matrix_t, matrix_t, matrix_t]:
    shape = matrix.shape
    assert shape[0] == shape[1]
    segment_size = shape[0] // 2
    a11 = matrix[0:segment_size, 0:segment_size]
    a12 = matrix[0:segment_size, segment_size:]
    a21 = matrix[segment_size:, 0:segment_size]
    a22 = matrix[segment_size:, segment_size:]

    return a11, a12, a21, a22


def add(matrix1: matrix_t, matrix2: matrix_t) -> matrix_t:
    float_op_cnt['+'] += matrix1.shape[0] * matrix1.shape[1]
    return matrix1 + matrix2


def sub(matrix1: matrix_t, matrix2: matrix_t) -> matrix_t:
    float_op_cnt['-'] += matrix1.shape[0] * matrix1.shape[1]
    return matrix1 - matrix2


def segment(matrix1: matrix_t, matrix2: matrix_t) \
        -> tuple[matrix_t, matrix_t, matrix_t, matrix_t, matrix_t, matrix_t, matrix_t]:
    a11, a12, a21, a22 = split_into_quarts(matrix1)
    b11, b12, b21, b22 = split_into_quarts(matrix2)

    m1 = strassen(add(a11, a22), add(b11, b22))
    m2 = strassen(add(a21, a22), b11)
    m3 = strassen(a11, sub(b12, b22))
    m4 = strassen(a22, sub(b21, b11))
    m5 = strassen(add(a11, a12), b22)
    m6 = strassen(sub(a21, a11), add(b11, b12))
    m7 = strassen(sub(a12, a22), add(b21, b22))

    return m1, m2, m3, m4, m5, m6, m7


def strassen(matrix1: matrix_t, matrix2: matrix_t):
    assert matrix1.shape == matrix2.shape
    assert matrix1.shape[0] == matrix1.shape[1]

    if matrix1.shape[0] == 1:
        float_op_cnt['*'] += 1
        return matrix1 * matrix2

    m1, m2, m3, m4, m5, m6, m7 = segment(matrix1, matrix2)

    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6

    upper = np.concatenate((c11, c12), axis=1)
    lower = np.concatenate((c21, c22), axis=1)

    return np.concatenate((upper, lower), axis=0)


if __name__ == '__main__':
    from numpy.random import rand
    import time

    # a = np.array([[1, 2], [3, 4]])
    # b = np.array([[10, 11], [14, 15]])
    2**1 - 2**10

    n = 64

    a = rand(n, n)
    b = rand(n, n)

    print((abs(strassen(a, b) - a@b) < 0.001).all())
    print(float_op_cnt)

    for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        float_op_cnt = {'*': 0, '/': 0, '+': 0, '-': 0}
        a = rand(n, n)
        b = rand(n, n)

        s = time.time()
        strassen(a, b)
        end = time.time()

        print('n =', n, ':', end - s)



