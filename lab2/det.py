from lu import lu
import numpy as np


def det(M : np.matrix):
    _det = 1
    _size, _ = M.shape
    _, U = lu(M)

    for i in range(_size):
        _det *= U[i, i]

    return _det


if __name__ == '__main__':
    N = 2 ** 3

    M = np.random.rand(N, N)

    print(np.linalg.det(M))
    print("---------------------------")
    print(det(M))