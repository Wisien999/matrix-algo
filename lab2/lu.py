import numpy as np
from scipy import linalg
from inverse import inverse
from copy import deepcopy


def lu(M: np.matrix):
    if M.shape == (1, 1):
        return np.matrix('1'), M
    else:
        _size, _ = M.shape
        M11 = M[0:_size//2, 0:_size//2]
        M12 = M[0:_size//2, _size//2:_size]
        M21 = M[_size//2:_size, 0:_size//2]
        M22 = M[_size//2:_size, _size//2:_size]

        # LU lewej górnej ćwiartki
        L11, U11 = lu(M11)

        # odwracanie U11
        inv_U11 = inverse(U11)

        # obliczanie L21
        L21 = np.matmul(M21, inv_U11)

        # odwracanie L11
        inv_L11 = inverse(L11)

        # obliczanie U12
        U12 = np.matmul(inv_L11, M12)

        # obliczanie L22
        L22 = M22 - (np.matmul(
            np.matmul(M21, inv_U11),
            np.matmul(inv_L11, M12)
        ))
        # obliczanie rekurencyjne L22 i U22
        L22, U22 = lu(L22)

        L = np.concatenate(
            (
                np.concatenate(
                    (L11, np.zeros(shape=(_size//2, _size//2))), axis=1),
                np.concatenate((L21, L22), axis=1)
            ),
            axis=0
        )

        U = np.concatenate(
            (
                np.concatenate((U11, U12), axis=1),
                np.concatenate(
                    (np.zeros(shape=(_size//2, _size//2)), U22), axis=1)
            ),
            axis=0
        )

        return L, U


if __name__ == '__main__':
    N = 2 ** 3

    M = np.random.rand(N, N)

    L, U = lu(M)

    P, L1, U1 = linalg.lu(M, check_finite=False)

    print("---------------------------")
    print(M)
    print("---------------------------")
    print(P @ L1 @ U1)
    if np.allclose(L @ U, M):
        print("Zgadza się!")
    else:
        print("No tak średnio")
