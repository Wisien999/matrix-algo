import numpy as np
from lab1.strassen import split_into_quarts, matrix_t


def inverse(matrix: matrix_t) -> matrix_t:
    if matrix.shape[0] == 1:
        return 1 / matrix

    a11, a12, a21, a22 = split_into_quarts(matrix)

    inv_a11 = inverse(a11)

    s22 = a22 - a21 @ inv_a11 @ a12

    inv_s22 = inverse(s22)

    b11 = inv_a11 @ (np.identity(inv_a11.shape[1]) + a12 @ inv_s22 @ a21 @ inv_a11)
    b12 = -inv_a11 @ a12 @ inv_s22
    b21 = - inv_s22 @ a21 @ inv_a11
    b22 = inv_s22

    return np.bmat([[b11, b12], [b21, b22]])


if __name__ == '__main__':
    A = np.array([[10.0, 40.0, 70.0, 32], [20.0, 50.0, 80.0, 12], [30.0, 60.0, 80.0, 4], [34, 64, 1, 1]])

    # A = np.array([[10.0, 40.0], [20.0, 50.0]])
    print(A)
    print('--------------------------')

    B = inverse(A)
    print(B)

    # print(A @ B)
    # print(B @ A)
    print(np.linalg.inv(A))
    print('--------------------------')
    print(A @ B)
    print(A @ np.linalg.inv(A))
